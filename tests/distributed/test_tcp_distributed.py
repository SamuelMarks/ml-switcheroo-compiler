"""Tests for test_tcp_distributed."""

from __future__ import annotations

import queue as py_queue
import typing

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from ml_switcheroo_compiler.backends.numpy.eager.distributed import (
    _np_all_gather,
    _np_all_reduce,
    _np_all_to_all,
    _np_reduce_scatter,
    set_np_distributed_context,
)
from tests.testing_utils import coerce_operand_list, require_callable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def _find_free_port_block(world_size: int) -> int:
    """Locate a contiguous block of free loopback TCP ports.

    Args:
        world_size (int): Number of contiguous ports required.

    Returns:
        int: Starting base port number.
    """
    import socket

    for candidate in range(38500, 39500, 10):
        socks = []
        ok = True
        for offset in range(world_size):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.bind(("127.0.0.1", candidate + offset))
                socks.append(sock)
            except OSError:
                ok = False
                break
        for sock in socks:
            sock.close()
        if ok:
            return candidate
    return 38500


def _worker_process(
    rank: int,
    world_size: int,
    port_queue: py_queue.Queue[int] | None,
    queue: py_queue.Queue[dict[str, BackendArg]],
) -> None:
    """Test function _worker_process.

    Args:
        rank: Test parameter `rank`.
        world_size: Test parameter `world_size`.
        port_queue: Test parameter `port_queue`.
        queue: Test parameter `queue`.
    """
    if port_queue is None:
        queue.put({"rank": rank, "error": "port_queue is required"})
        return
    try:
        if rank == 0:
            import socket

            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("", 0))
            port = int(s.getsockname()[1])
            s.close()
            port_queue.put(port)
        else:
            port = int(port_queue.get(timeout=15))
            port_queue.put(port)
        set_np_distributed_context(world_size, rank, "127.0.0.1", port)
        import ml_switcheroo_compiler.backends.numpy.eager.distributed as dmod

        tensor = np.array([rank], dtype=np.float32)
        gathered = _np_all_gather(np, tensor, axis=0)
        reduced_sum = _np_all_reduce(np, tensor, op_type="sum")
        reduced_max = _np_all_reduce(np, tensor, op_type="max")
        tensor_rs = np.arange(world_size, dtype=np.float32) * (rank + 1)
        scattered = _np_reduce_scatter(np, tensor_rs, op_type="sum", axis=0)
        a2a = _np_all_to_all(np, tensor)
        dmod._tcp_dist_ctx.shutdown()
        queue.put(
            {
                "rank": rank,
                "gathered": gathered,
                "reduced_sum": reduced_sum,
                "reduced_max": reduced_max,
                "scattered": scattered,
                "a2a_len": len(coerce_operand_list(a2a)),
            }
        )
    except Exception as e:
        queue.put({"rank": rank, "error": str(e)})


def test_tcp_distributed_collectives() -> None:
    """Test function test_tcp_distributed_collectives."""
    import threading

    from ml_switcheroo_compiler.backends.numpy.eager.distributed import TCPDistributedContext

    # 1. Verify _worker_process error capture and rank-0/non-zero port handoff paths
    err_q: py_queue.Queue[dict[str, BackendArg]] = py_queue.Queue()
    _worker_process(1, 2, None, err_q)
    err_item = err_q.get(timeout=2)
    assert err_item["rank"] == 1 and "error" in err_item

    pq1: py_queue.Queue[int] = py_queue.Queue()
    rq1: py_queue.Queue[dict[str, BackendArg]] = py_queue.Queue()
    _worker_process(0, 1, pq1, rq1)
    r0_res = rq1.get(timeout=2)
    assert "error" not in r0_res and r0_res["rank"] == 0

    pq2: py_queue.Queue[int] = py_queue.Queue()
    pq2.put(pq1.get(timeout=2))
    rq2: py_queue.Queue[dict[str, BackendArg]] = py_queue.Queue()
    _worker_process(1, 1, pq2, rq2)
    r1_res = rq2.get(timeout=2)
    assert "error" not in r1_res and r1_res["rank"] == 1

    # 2. Multi-rank ring collectives over real loopback TCP sockets across 3 threads
    world_size = 3
    base_port = _find_free_port_block(world_size)
    thread_results: dict[int, dict[str, BackendArg]] = {}

    def _run_ring_rank(rank: int) -> None:
        """Execute real 3-rank TCP ring collectives for `rank`.

        Args:
            rank (int): Worker rank index.
        """
        ctx = TCPDistributedContext(world_size=world_size, rank=rank, addr="127.0.0.1", port=base_port)
        ctx.initialize()
        try:
            vec = np.array([float(rank), float(rank + 1), float(rank + 2)], dtype=np.float64)
            gathered = ctx.all_gather_tensors(np.array([rank], dtype=np.float32))
            red_sum = ctx.all_reduce_ring(vec, op_type="sum")
            red_max = ctx.all_reduce_ring(vec, op_type="max")
            thread_results[rank] = {"gathered": gathered, "sum": red_sum, "max": red_max}
        finally:
            ctx.shutdown()

    threads = [threading.Thread(target=_run_ring_rank, args=(r,)) for r in range(world_size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert len(thread_results) == world_size
    for r in range(world_size):
        res_r = thread_results[r]
        gathered_list = [ensure_ndarray(x).item() for x in coerce_operand_list(res_r["gathered"])]
        assert gathered_list == [0.0, 1.0, 2.0]
        np.testing.assert_allclose(ensure_ndarray(res_r["sum"]), np.array([3.0, 6.0, 9.0]))
        np.testing.assert_allclose(ensure_ndarray(res_r["max"]), np.array([2.0, 3.0, 4.0]))


def test_tcp_distributed_single_node() -> None:
    """Test function test_tcp_distributed_single_node."""
    set_np_distributed_context(1, 0, "127.0.0.1", 29506)
    tensor = np.array([5.0])
    assert ensure_ndarray(_np_all_reduce(np, tensor)).item() == 5.0
    assert ensure_ndarray(_np_all_gather(np, tensor)).item() == 5.0
    assert ensure_ndarray(_np_reduce_scatter(np, tensor)).item() == 5.0


def test_tcp_distributed_missing_funcs() -> None:
    """Test function test_tcp_distributed_missing_funcs."""
    import ml_switcheroo_compiler.backends.numpy.eager.distributed as dmod

    dmod._tcp_dist_ctx.shutdown()
    import random

    from ml_switcheroo_compiler.backends.numpy.eager.distributed import (
        _np_all_reduce,
        _np_broadcast,
        _np_reduce,
        _np_reduce_scatter,
        _tcp_dist_ctx,
    )

    random.randint(40000, 60000)
    _tcp_dist_ctx.world_size = 2
    _tcp_dist_ctx.rank = 0
    tensor = np.array([2.0])

    def mock_all_gather(t: BackendArg) -> BackendArg:
        """Test function mock_all_gather.

        Args:
            t: Test parameter `t`.

        Returns:
            BackendArg: Computed output value from `mock_all_gather`.
        """
        return [t, np.array([3.0])]

    pytest.MonkeyPatch().setattr(_tcp_dist_ctx, "all_gather_tensors", mock_all_gather, raising=False)
    pytest.MonkeyPatch().setattr(
        _tcp_dist_ctx,
        "all_reduce_ring",
        lambda t, op, bm: np.array([6.0]) if op == "prod" else np.array([5.0]) if op == "unknown" else t,
        raising=False,
    )
    _tcp_dist_ctx.world_size = 2
    _tcp_dist_ctx.world_size = 2
    assert ensure_ndarray(_np_all_reduce(np, tensor, op_type="prod")).item() == 6.0
    assert ensure_ndarray(_np_all_reduce(np, tensor, op_type="min")).item() == 2.0
    assert ensure_ndarray(_np_all_reduce(np, tensor, op_type="unknown")).item() == 5.0
    assert ensure_ndarray(_np_reduce_scatter(np, tensor, op_type="prod", axis=0)).item() == 6.0
    assert ensure_ndarray(_np_reduce_scatter(np, tensor, op_type="max", axis=0)).item() == 3.0
    assert ensure_ndarray(_np_reduce_scatter(np, tensor, op_type="min", axis=0)).item() == 2.0
    assert ensure_ndarray(_np_reduce_scatter(np, tensor, op_type="unknown", axis=0)).item() == 5.0
    assert ensure_ndarray(_np_reduce(np, tensor, root_rank=0, op_type="prod")).item() == 6.0
    assert ensure_ndarray(_np_reduce(np, tensor, root_rank=0, op_type="max")).item() == 3.0
    assert ensure_ndarray(_np_reduce(np, tensor, root_rank=0, op_type="min")).item() == 2.0
    assert ensure_ndarray(_np_reduce(np, tensor, root_rank=0, op_type="unknown")).item() == 5.0
    _tcp_dist_ctx.rank = 1
    assert _np_reduce(np, tensor, root_rank=0, op_type="prod") is None
    assert ensure_ndarray(_np_broadcast(np, tensor, root_rank=0)).item() == 2.0


def test_tcp_distributed_context_none_connections() -> None:
    """Test function test_tcp_distributed_context_none_connections."""
    from unittest.mock import patch

    import numpy as np

    from ml_switcheroo_compiler.backends.numpy.eager.distributed import TCPDistributedContext

    with patch("threading.Thread"):
        with patch("ml_switcheroo_compiler.backends.numpy.eager.distributed.Listener") as mock_listener:
            mock_listener.return_value = None
            with patch(
                "ml_switcheroo_compiler.backends.numpy.eager.distributed.Client", side_effect=ConnectionRefusedError
            ):
                ctx = TCPDistributedContext(world_size=2, rank=0)
                ctx.listener = None
                ctx.send_conns = []
                ctx.recv_conns = []
                tensor = np.array([1.0, 2.0])
                out = ctx.all_reduce_ring(tensor)
                assert out.shape == (2,)


def test_tcp_distributed_context_none_connections_all_reduce() -> None:
    """Test function test_tcp_distributed_context_none_connections_all_reduce."""
    import numpy as np

    from ml_switcheroo_compiler.backends.numpy.eager.distributed import TCPDistributedContext

    class DummyConn:
        """Test class DummyConn."""

        def recv(self) -> BackendArg:
            """Test function recv.

            Returns:
                BackendArg: Computed output value from `recv`.
            """
            return np.array([0])

        def send(self, data: BackendArg) -> None:
            """Test function send.

            Args:
                data: Test parameter `data`.
            """
            pass

    ctx = TCPDistributedContext(world_size=2, rank=0)
    ctx.listener = None
    pytest.MonkeyPatch().setattr(ctx, "send_conns", [DummyConn()], raising=False)
    pytest.MonkeyPatch().setattr(ctx, "recv_conns", [DummyConn()], raising=False)
    tensor = np.array([1, 2])
    ctx.all_reduce_ring(tensor)
    assert tuple(tensor.shape) == (2,)


def test_tcp_distributed_context_none_listener_thread() -> None:
    """Test function test_tcp_distributed_context_none_listener_thread."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.backends.numpy.eager.distributed import TCPDistributedContext

    with patch("ml_switcheroo_compiler.backends.numpy.eager.distributed.Listener") as mock_listener:
        mock_listener.return_value = None
        with patch(
            "ml_switcheroo_compiler.backends.numpy.eager.distributed.Client", side_effect=ConnectionRefusedError
        ):
            ctx = TCPDistributedContext(world_size=2, rank=0)
            assert ctx.world_size == 2


def test_tcp_distributed_context_none_listener_thread_fixed() -> None:
    """Test function test_tcp_distributed_context_none_listener_thread_fixed."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.backends.numpy.eager.distributed import TCPDistributedContext

    with patch("ml_switcheroo_compiler.backends.numpy.eager.distributed.Listener", side_effect=lambda *a, **kw: None):
        with patch(
            "ml_switcheroo_compiler.backends.numpy.eager.distributed.Client", side_effect=ConnectionRefusedError
        ):
            ctx = TCPDistributedContext(world_size=2, rank=0)
            assert ctx.world_size == 2


def test_tcp_distributed_context_sync_thread() -> None:
    """Test function test_tcp_distributed_context_sync_thread."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.backends.numpy.eager.distributed import TCPDistributedContext

    class SyncThread:
        """Test class SyncThread."""

        def __init__(self, target: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> None:
            """Test function __init__.

            Args:
                target: Test parameter `target`.
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            self.target = target

        def start(self) -> None:
            """Test function start."""
            require_callable(self.target)()

        def join(self) -> None:
            """Test function join."""
            pass

    with patch("threading.Thread", new=SyncThread):
        with patch(
            "ml_switcheroo_compiler.backends.numpy.eager.distributed.Listener", side_effect=lambda *a, **kw: None
        ):
            with patch(
                "ml_switcheroo_compiler.backends.numpy.eager.distributed.Client", side_effect=ConnectionRefusedError
            ):
                ctx = TCPDistributedContext(world_size=2, rank=0)
                assert ctx.world_size == 2


def test_tcp_distributed_context_initialize_none_listener() -> None:
    """Test function test_tcp_distributed_context_initialize_none_listener."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.backends.numpy.eager.distributed import TCPDistributedContext

    class SyncThread:
        """Test class SyncThread."""

        def __init__(self, target: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> None:
            """Test function __init__.

            Args:
                target: Test parameter `target`.
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            self.target = target

        def start(self) -> None:
            """Test function start."""
            require_callable(self.target)()

        def join(self) -> None:
            """Test function join."""
            pass

    with patch("threading.Thread", new=SyncThread):
        with patch("time.sleep", return_value=None):
            with patch(
                "ml_switcheroo_compiler.backends.numpy.eager.distributed.Listener", side_effect=lambda *a, **kw: None
            ):
                with patch(
                    "ml_switcheroo_compiler.backends.numpy.eager.distributed.Client", side_effect=ConnectionRefusedError
                ):
                    ctx = TCPDistributedContext(world_size=2, rank=0)
                    ctx.initialize()
                    assert ctx.listener is None
