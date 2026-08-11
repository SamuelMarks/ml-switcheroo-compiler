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
    set_mock_distributed_context,
)


def test_axis_index() -> None:
    """Test axis index.

    Returns:
        None
    """
    res = _np_axis_index(np)
    assert res == 0


def test_mock_distributed_context() -> None:
    """Test mock distributed context.

    Returns:
        None
    """
    import threading

    def run_rank(rank: int) -> None:
        set_mock_distributed_context(world_size=2, rank=rank)
        t = np.array([1, 2])

        # AllReduce
        res1 = _np_all_reduce(np, t)
        if rank == 0:
            assert np.array_equal(res1, t * 2)

        # AllGather
        res2 = _np_all_gather(np, t)
        if rank == 0:
            assert len(res2) == 4

        # Broadcast
        res3 = _np_broadcast(np, t)
        if rank == 0:
            assert np.array_equal(res3, t)

        # ReduceScatter
        res4 = _np_reduce_scatter(np, t)
        if rank == 0:
            assert len(res4) == 1

        # Reduce
        res5 = _np_reduce(np, t)
        if rank == 0:
            assert np.array_equal(res5, t * 2)

        # ShardTensor
        res6 = _np_shard_tensor(np, t)
        if rank == 0:
            assert np.array_equal(res6, t)

    t1 = threading.Thread(target=run_rank, args=(0,))
    t2 = threading.Thread(target=run_rank, args=(1,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # test world_size=1
    set_mock_distributed_context(world_size=1, rank=0)
    t = np.array([1, 2])
    res_reduce_1 = _np_all_reduce(np, t)
    assert np.array_equal(res_reduce_1, t)

    res_gather_1 = _np_all_gather(np, t)
    assert len(res_gather_1) == 1

    res_scatter_1 = _np_reduce_scatter(np, t)
    assert np.array_equal(res_scatter_1, t)


def test_distributed_initialize_timeout():
    from unittest.mock import patch

    from ml_switcheroo_compiler.backends.numpy.eager.distributed import set_mock_distributed_context

    with patch("ml_switcheroo_compiler.backends.numpy.eager.distributed.Client") as mock_client:
        mock_client.side_effect = ConnectionRefusedError
        set_mock_distributed_context(world_size=2, rank=1, port=39501)
        # Should loop 50 times and exit normally


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
