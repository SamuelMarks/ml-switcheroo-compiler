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
