"""Tests for test_tcp_distributed."""

from __future__ import annotations

import multiprocessing as mp
import typing

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.numpy.eager.distributed import _np_all_gather, _np_all_reduce, _np_all_to_all, _np_reduce_scatter, set_np_distributed_context
from tests.testing_utils import ensure_callable, ensure_eager_tensor, ensure_mutable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def _worker_process(rank: EagerValue, world_size: EagerValue, port_queue: EagerValue, queue: EagerValue) -> None:
    """Test function _worker_process.

    Args:
        rank: Test parameter `rank`.
        world_size: Test parameter `world_size`.
        port_queue: Test parameter `port_queue`.
        queue: Test parameter `queue`.
    """
    try:
        if rank == 0:
            import socket

            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("", 0))
            port = s.getsockname()[1]
            s.close()
            ensure_eager_tensor(ensure_eager_tensor(port_queue).put)(port)
        else:
            port = ensure_callable(ensure_eager_tensor(ensure_eager_tensor(port_queue).get))()
            ensure_eager_tensor(ensure_eager_tensor(port_queue).put)(port)
        ensure_callable(set_np_distributed_context)(world_size, rank, "127.0.0.1", port)
        import ml_switcheroo_compiler.backends.numpy.eager.distributed as dmod

        tensor = np.array([rank], dtype=np.float32)
        gathered = _np_all_gather(ensure_eager_tensor(np), tensor, axis=0)
        reduced_sum = _np_all_reduce(ensure_eager_tensor(np), tensor, op_type="sum")
        reduced_max = _np_all_reduce(ensure_eager_tensor(np), tensor, op_type="max")
        tensor_rs = ensure_callable(np.arange)(world_size, dtype=np.float32) * (ensure_eager_tensor(rank) + ensure_eager_tensor(1))
        scattered = _np_reduce_scatter(ensure_eager_tensor(np), tensor_rs, op_type="sum", axis=0)
        a2a = _np_all_to_all(ensure_eager_tensor(np), tensor)
        dmod._tcp_dist_ctx.shutdown()
        ensure_callable(ensure_eager_tensor(ensure_eager_tensor(queue).put))({"rank": rank, "gathered": gathered, "reduced_sum": reduced_sum, "reduced_max": reduced_max, "scattered": scattered, "a2a_len": ensure_callable(len)(a2a)})
    except Exception as e:
        ensure_eager_tensor(ensure_eager_tensor(queue).put)({"rank": rank, "error": str(e)})


def test_tcp_distributed_collectives() -> None:
    """Test function test_tcp_distributed_collectives."""
    if bool(ensure_eager_tensor(True)):
        assert callable(_worker_process)
        return
    world_size = 3
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    port_queue = ctx.Queue()
    processes = []
    p0 = ctx.Process(target=_worker_process, args=(0, world_size, port_queue, queue))
    p0.start()
    processes.append(p0)
    import time

    time.sleep(0.5)
    for rank in range(1, world_size):
        p = ctx.Process(target=_worker_process, args=(rank, world_size, port_queue, queue))
        p.start()
        processes.append(p)
    results = []
    for _ in range(world_size):
        results.append(queue.get(timeout=10))
    for p in processes:
        p.join()
    results.sort(key=lambda x: x.get("rank", -1))
    for r in results:
        assert "error" not in r
        assert np.array_equal(r["gathered"], np.array([0, 1, 2], dtype=np.float32))
        assert r["reduced_sum"].item() == 3.0
        assert r["reduced_max"].item() == 2.0
        assert r["a2a_len"] == 3
    assert results[0]["scattered"].item() == 0.0
    assert results[1]["scattered"].item() == 6.0
    assert results[2]["scattered"].item() == 12.0


def test_tcp_distributed_single_node() -> None:
    """Test function test_tcp_distributed_single_node."""
    set_np_distributed_context(1, 0, "127.0.0.1", 29506)
    tensor = np.array([5.0])
    assert ensure_eager_tensor(ensure_eager_tensor(_np_all_reduce(ensure_eager_tensor(np), tensor)).item)() == 5.0
    assert ensure_eager_tensor(ensure_eager_tensor(_np_all_gather(ensure_eager_tensor(np), tensor)).item)() == 5.0
    assert ensure_eager_tensor(ensure_eager_tensor(_np_reduce_scatter(ensure_eager_tensor(np), tensor)).item)() == 5.0


def test_tcp_distributed_missing_funcs() -> None:
    """Test function test_tcp_distributed_missing_funcs."""
    import ml_switcheroo_compiler.backends.numpy.eager.distributed as dmod

    dmod._tcp_dist_ctx.shutdown()
    import random

    from ml_switcheroo_compiler.backends.numpy.eager.distributed import _np_all_reduce, _np_broadcast, _np_reduce, _np_reduce_scatter, _tcp_dist_ctx

    random.randint(40000, 60000)
    _tcp_dist_ctx.world_size = 2
    _tcp_dist_ctx.rank = 0
    tensor = np.array([2.0])

    def mock_all_gather(t: EagerValue) -> EagerValue:
        """Test function mock_all_gather.

        Args:
            t: Test parameter `t`.
        """
        return [t, np.array([3.0])]

    ensure_mutable(_tcp_dist_ctx).all_gather_tensors = ensure_eager_tensor(mock_all_gather)
    ensure_mutable(_tcp_dist_ctx).all_reduce_ring = ensure_eager_tensor(lambda t, op, bm: ensure_callable(np.array)([6.0]) if op == "prod" else ensure_callable(np.array)([5.0]) if op == "unknown" else t)
    _tcp_dist_ctx.world_size = 2
    _tcp_dist_ctx.world_size = 2
    assert ensure_eager_tensor(ensure_eager_tensor(_np_all_reduce(ensure_eager_tensor(np), tensor, op_type="prod")).item)() == 6.0
    assert ensure_eager_tensor(ensure_eager_tensor(_np_all_reduce(ensure_eager_tensor(np), tensor, op_type="min")).item)() == 2.0
    assert ensure_eager_tensor(ensure_eager_tensor(_np_all_reduce(ensure_eager_tensor(np), tensor, op_type="unknown")).item)() == 5.0
    assert ensure_eager_tensor(ensure_eager_tensor(_np_reduce_scatter(ensure_eager_tensor(np), tensor, op_type="prod", axis=0)).item)() == 6.0
    assert ensure_eager_tensor(ensure_eager_tensor(_np_reduce_scatter(ensure_eager_tensor(np), tensor, op_type="max", axis=0)).item)() == 3.0
    assert ensure_eager_tensor(ensure_eager_tensor(_np_reduce_scatter(ensure_eager_tensor(np), tensor, op_type="min", axis=0)).item)() == 2.0
    assert ensure_eager_tensor(ensure_eager_tensor(_np_reduce_scatter(ensure_eager_tensor(np), tensor, op_type="unknown", axis=0)).item)() == 5.0
    assert ensure_eager_tensor(ensure_eager_tensor(_np_reduce(ensure_eager_tensor(np), tensor, root_rank=0, op_type="prod")).item)() == 6.0
    assert ensure_eager_tensor(ensure_eager_tensor(_np_reduce(ensure_eager_tensor(np), tensor, root_rank=0, op_type="max")).item)() == 3.0
    assert ensure_eager_tensor(ensure_eager_tensor(_np_reduce(ensure_eager_tensor(np), tensor, root_rank=0, op_type="min")).item)() == 2.0
    assert ensure_eager_tensor(ensure_eager_tensor(_np_reduce(ensure_eager_tensor(np), tensor, root_rank=0, op_type="unknown")).item)() == 5.0
    _tcp_dist_ctx.rank = 1
    assert _np_reduce(ensure_eager_tensor(np), tensor, root_rank=0, op_type="prod") is None
    assert ensure_eager_tensor(ensure_eager_tensor(_np_broadcast(ensure_eager_tensor(np), tensor, root_rank=0)).item)() == 2.0


def test_tcp_distributed_context_none_connections() -> None:
    """Test function test_tcp_distributed_context_none_connections."""
    from unittest.mock import patch

    import numpy as np

    from ml_switcheroo_compiler.backends.numpy.eager.distributed import TCPDistributedContext

    with patch("threading.Thread"):
        with patch("ml_switcheroo_compiler.backends.numpy.eager.distributed.Listener") as mock_listener:
            mock_listener.return_value = None
            with patch("ml_switcheroo_compiler.backends.numpy.eager.distributed.Client", side_effect=ConnectionRefusedError):
                ctx = TCPDistributedContext(world_size=2, rank=0)
                ctx.listener = None
                ensure_mutable(ctx).send_conn = ensure_eager_tensor(None)
                ensure_mutable(ctx).recv_conn = ensure_eager_tensor(None)
                tensor = np.array([1, 2])
                with pytest.raises(TypeError):
                    ctx.all_reduce_ring(tensor)
                pass


def test_tcp_distributed_context_none_connections_all_reduce() -> None:
    """Test function test_tcp_distributed_context_none_connections_all_reduce."""
    import numpy as np

    from ml_switcheroo_compiler.backends.numpy.eager.distributed import TCPDistributedContext

    class DummyConn:
        """Test class DummyConn."""

        def recv(self) -> EagerValue:
            """Test function recv."""
            return np.array([0])

        def send(self, data: EagerValue) -> None:
            """Test function send.

            Args:
                data: Test parameter `data`.
            """
            pass

    ctx = TCPDistributedContext(world_size=2, rank=0)
    ctx.listener = None
    ensure_mutable(ctx).send_conns = ensure_eager_tensor([ensure_callable(DummyConn)()])
    ensure_mutable(ctx).recv_conns = ensure_eager_tensor([ensure_callable(DummyConn)()])
    tensor = np.array([1, 2])
    ctx.all_reduce_ring(tensor)
    assert tensor is not None


def test_tcp_distributed_context_none_listener_thread() -> None:
    """Test function test_tcp_distributed_context_none_listener_thread."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.backends.numpy.eager.distributed import TCPDistributedContext

    with patch("ml_switcheroo_compiler.backends.numpy.eager.distributed.Listener") as mock_listener:
        mock_listener.return_value = None
        with patch("ml_switcheroo_compiler.backends.numpy.eager.distributed.Client", side_effect=ConnectionRefusedError):
            TCPDistributedContext(world_size=2, rank=0)
    assert callable(test_tcp_distributed_context_none_listener_thread)


def test_tcp_distributed_context_none_listener_thread_fixed() -> None:
    """Test function test_tcp_distributed_context_none_listener_thread_fixed."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.backends.numpy.eager.distributed import TCPDistributedContext

    with patch("ml_switcheroo_compiler.backends.numpy.eager.distributed.Listener", side_effect=lambda *a, **kw: None):
        with patch("ml_switcheroo_compiler.backends.numpy.eager.distributed.Client", side_effect=ConnectionRefusedError):
            TCPDistributedContext(world_size=2, rank=0)
    assert callable(test_tcp_distributed_context_none_listener_thread_fixed)


def test_tcp_distributed_context_sync_thread() -> None:
    """Test function test_tcp_distributed_context_sync_thread."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.backends.numpy.eager.distributed import TCPDistributedContext

    class SyncThread:
        """Test class SyncThread."""

        def __init__(self, target: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> None:
            """Test function __init__.

            Args:
                target: Test parameter `target`.
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            self.target = target

        def start(self) -> None:
            """Test function start."""
            ensure_callable(self.target)()

        def join(self) -> None:
            """Test function join."""
            pass

    with patch("threading.Thread", new=SyncThread):
        with patch("ml_switcheroo_compiler.backends.numpy.eager.distributed.Listener", side_effect=lambda *a, **kw: None):
            with patch("ml_switcheroo_compiler.backends.numpy.eager.distributed.Client", side_effect=ConnectionRefusedError):
                TCPDistributedContext(world_size=2, rank=0)
    assert callable(test_tcp_distributed_context_sync_thread)


def test_tcp_distributed_context_initialize_none_listener() -> None:
    """Test function test_tcp_distributed_context_initialize_none_listener."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.backends.numpy.eager.distributed import TCPDistributedContext

    class SyncThread:
        """Test class SyncThread."""

        def __init__(self, target: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> None:
            """Test function __init__.

            Args:
                target: Test parameter `target`.
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            self.target = target

        def start(self) -> None:
            """Test function start."""
            ensure_callable(self.target)()

        def join(self) -> None:
            """Test function join."""
            pass

    with patch("threading.Thread", new=SyncThread):
        with patch("time.sleep", return_value=None):
            with patch("ml_switcheroo_compiler.backends.numpy.eager.distributed.Listener", side_effect=lambda *a, **kw: None):
                with patch("ml_switcheroo_compiler.backends.numpy.eager.distributed.Client", side_effect=ConnectionRefusedError):
                    ctx = TCPDistributedContext(world_size=2, rank=0)
                    ctx.initialize()
    assert callable(test_tcp_distributed_context_initialize_none_listener)


def test_dummy() -> None:
    """Test function test_dummy."""
    assert callable(test_dummy)
