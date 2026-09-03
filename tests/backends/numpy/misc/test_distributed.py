"""Tests for numpy eager distributed ops."""

import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.distributed import (
    _np_all_gather,
    _np_all_reduce,
    _np_axis_index,
    _np_broadcast,
    _np_reduce,
    _np_reduce_scatter,
    _np_shard_tensor,
    set_np_distributed_context,
)


def test_axis_index() -> None:
    """Test axis index.

    Returns:
        None
    """
    res = _np_axis_index(np)
    assert res == 0


def test_np_distributed_context() -> None:
    """Test mock distributed context."""
    from unittest.mock import MagicMock, patch

    from ml_switcheroo_compiler.backends.numpy.eager.distributed import _np_all_to_all, _tcp_dist_ctx

    mock_conn = MagicMock()
    mock_conn.recv.return_value = np.array([3, 4])

    with patch("ml_switcheroo_compiler.backends.numpy.eager.distributed.Listener") as MockListener:
        with patch("ml_switcheroo_compiler.backends.numpy.eager.distributed.Client", return_value=mock_conn):
            with patch("time.sleep"):
                mock_listener_instance = MagicMock()
                mock_listener_instance.accept.return_value = mock_conn
                MockListener.return_value = mock_listener_instance

                set_np_distributed_context(world_size=2, rank=0)
                t = np.array([1, 2])

                res1 = _np_all_reduce(np, t)
                res2 = _np_all_gather(np, t)
                res3 = _np_broadcast(np, t)
                res4 = _np_reduce_scatter(np, t)
                res5 = _np_reduce(np, t)
                res6 = _np_shard_tensor(np, t)

                # test reduce scatter ops
                _np_reduce_scatter(np, t, "prod")
                _np_reduce_scatter(np, t, "max")
                _np_reduce_scatter(np, t, "min")
                _np_reduce_scatter(np, t, "unknown")

                # test reduce ops
                _np_reduce(np, t, 0, "prod")
                _np_reduce(np, t, 0, "max")
                _np_reduce(np, t, 0, "min")
                _np_reduce(np, t, 0, "unknown")

                # test all reduce ops
                _np_all_reduce(np, t, "prod")
                _np_all_reduce(np, t, "max")
                _np_all_reduce(np, t, "min")
                _np_all_reduce(np, t, "unknown")

                # test all to all
                _np_all_to_all(np, t)

                # also test the underlying ring
                _tcp_dist_ctx.all_reduce_ring(t, op_type="prod")
                _tcp_dist_ctx.all_reduce_ring(t, op_type="max")
                _tcp_dist_ctx.all_reduce_ring(t, op_type="min")

                _tcp_dist_ctx.shutdown()

    # test world_size=1
    set_np_distributed_context(world_size=1, rank=0)
    t = np.array([1, 2])
    res_reduce_1 = _np_all_reduce(np, t)
    res_gather_1 = _np_all_gather(np, t)
    res_scatter_1 = _np_reduce_scatter(np, t)

    assert np.array_equal(_np_all_to_all(np, t), t)
    assert np.array_equal(_np_reduce(np, t), t)
    assert np.array_equal(_np_shard_tensor(np, t), t)
    assert np.array_equal(_np_broadcast(np, t), t)
    assert np.array_equal(_tcp_dist_ctx.all_reduce_ring(t), t)
    assert np.array_equal(_tcp_dist_ctx.all_gather_tensors(t)[0], t)


def test_distributed_initialize_timeout():
    from unittest.mock import MagicMock, patch

    from ml_switcheroo_compiler.backends.numpy.eager.distributed import set_np_distributed_context

    with patch("ml_switcheroo_compiler.backends.numpy.eager.distributed.Client") as mock_client:
        with patch("ml_switcheroo_compiler.backends.numpy.eager.distributed.Listener") as MockListener:
            with patch("time.sleep"):
                mock_listener_instance = MagicMock()
                mock_listener_instance.accept.return_value = MagicMock()
                MockListener.return_value = mock_listener_instance

                mock_client.side_effect = ConnectionRefusedError
                set_np_distributed_context(world_size=2, rank=1, port=39501)


def test_distributed_shutdown_no_conn():
    from ml_switcheroo_compiler.backends.numpy.eager.distributed import TCPDistributedContext

    ctx = TCPDistributedContext()
    ctx.world_size = 2
    ctx.rank = 1
    ctx.conn = None
    ctx.shutdown()

    ctx2 = TCPDistributedContext()
    ctx2.world_size = 2
    ctx2.rank = 0
    ctx2.connections = []
    ctx2.listener = None
    ctx2.shutdown()


def test_ipc_timeout():
    from ml_switcheroo_compiler.backends.numpy.distributed.ipc import _exchange_ipc_data_worker

    def mock_target(rank, data):
        pass

    try:
        _exchange_ipc_data_worker(1, 2, np.array([1]), 0.001, 0.001)
    except TimeoutError:
        pass
    except Exception:
        pass


def test_ipc_worker_retry_then_success():
    from unittest.mock import MagicMock, patch

    import numpy as np

    from ml_switcheroo_compiler.backends.numpy.distributed.ipc import _exchange_ipc_data_worker

    class MockClient:
        def __init__(self, *args, **kwargs):
            self.calls = getattr(self.__class__, "calls", 0)
            self.__class__.calls = self.calls + 1
            if self.__class__.calls < 2:
                raise ConnectionRefusedError("fail first time")
            self.mock = MagicMock()

        def __enter__(self):
            return self.mock

        def __exit__(self, *args):
            pass

    with patch("ml_switcheroo_compiler.backends.numpy.distributed.ipc.Client", new=MockClient):
        with patch("ml_switcheroo_compiler.backends.numpy.distributed.ipc.Listener") as MockListener:
            mock_listener = MagicMock()
            mock_listener.__enter__.return_value = mock_listener
            mock_listener.accept.return_value.__enter__.return_value.recv.return_value = [np.array([1]), np.array([2])]
            MockListener.return_value = mock_listener

            _exchange_ipc_data_worker(1, 2, np.array([1]), 1.0, 0.001)


from unittest.mock import MagicMock

import ml_switcheroo_compiler.backends.numpy.eager.distributed as dmod
from ml_switcheroo_compiler.backends.numpy.eager.distributed import TCPDistributedContext, _np_all_to_all


def test_tcp_dist_ctx_coverage():
    # Test world_size <= 1 return [tensor]
    ctx = TCPDistributedContext(world_size=1)
    assert ctx.all_gather_tensors("test") == ["test"]

    # Test rank 0 gather
    ctx = TCPDistributedContext(world_size=2, rank=0)
    mock_conn = MagicMock()
    mock_conn.recv.return_value = "B"
    ctx.recv_conns = [mock_conn]
    ctx.send_conns = [mock_conn]
    res = ctx.all_gather_tensors("A")
    assert res == ["A", "B"]
    mock_conn.send.assert_called_with("A")

    # Test rank 0 shutdown
    ctx.listener = MagicMock()
    ctx.shutdown()
    assert mock_conn.close.call_count == 2
    ctx.listener.close.assert_called_once()

    # Test rank 1 missing conn
    ctx = TCPDistributedContext(world_size=2, rank=1)
    res = ctx.all_gather_tensors("A")
    assert res == [None, "A"]


def test_np_ops_coverage():
    dmod._tcp_dist_ctx.world_size = 2
    dmod._tcp_dist_ctx.rank = 0
    dmod._tcp_dist_ctx.all_reduce_ring = lambda t, op, bm: np.array([5.0]) if op == "sum" else np.array([3.0])
    dmod._tcp_dist_ctx.all_gather_tensors = lambda t: [t, np.array([3.0])]

    t = np.array([2.0])
    # cover sum for all ops
    assert _np_all_reduce(np, t, op_type="sum").item() == 5.0

    dmod._tcp_dist_ctx.all_gather_tensors = lambda t: [t, np.array([3.0])]
    assert _np_reduce(np, t, op_type="sum").item() == 5.0

    # cover max for all ops
    assert _np_all_reduce(np, t, op_type="max").item() == 3.0

    # cover all_gather concatenate
    res = _np_all_gather(np, t, axis=0)
    assert np.array_equal(res, [2.0, 3.0])

    # cover reduce_scatter sum
    t_rs = np.array([1.0, 2.0])
    dmod._tcp_dist_ctx.all_gather_tensors = lambda t: [t, np.array([3.0, 4.0])]
    assert _np_reduce_scatter(np, t_rs, op_type="sum").item() == 4.0

    # cover all_to_all
    a2a = _np_all_to_all(np, t_rs)
    assert len(a2a) == 2


from unittest.mock import patch


def test_tcp_dist_ctx_connection_refused():
    ctx = TCPDistributedContext(world_size=2, rank=1)

    with patch("ml_switcheroo_compiler.backends.numpy.eager.distributed.Client") as mock_client:
        with patch("ml_switcheroo_compiler.backends.numpy.eager.distributed.Listener") as MockListener:
            with patch("time.sleep"):
                mock_listener_instance = MagicMock()
                mock_listener_instance.accept.return_value = MagicMock()
                MockListener.return_value = mock_listener_instance

                # First call raises ConnectionRefusedError, second returns a mock connection
                mock_client.side_effect = [ConnectionRefusedError(), MagicMock()]
                ctx.initialize()

                assert mock_client.call_count == 2
                assert len(ctx.send_conns) > 0
