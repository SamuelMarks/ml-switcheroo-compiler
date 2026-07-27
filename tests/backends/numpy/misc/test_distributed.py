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
    set_mock_distributed_context(world_size=2, rank=1)

    t = np.array([1, 2])

    # AllReduce
    res1 = _np_all_reduce(np, t)
    assert np.array_equal(res1, t)

    # AllGather
    res2 = _np_all_gather(np, t)
    assert len(res2) == 4

    # Broadcast
    res3 = _np_broadcast(np, t)
    assert np.array_equal(res3, t)

    # ReduceScatter
    res4 = _np_reduce_scatter(np, t)
    assert np.array_equal(res4, t)

    # Reduce
    res5 = _np_reduce(np, t)
    assert np.array_equal(res5, t)

    # ShardTensor
    res6 = _np_shard_tensor(np, t)
    assert np.array_equal(res6, t)

    # test world_size=1
    set_mock_distributed_context(world_size=1, rank=0)
    res_reduce_1 = _np_all_reduce(np, t)
    assert np.array_equal(res_reduce_1, t)

    res_gather_1 = _np_all_gather(np, t)
    assert len(res_gather_1) == 1

    res_scatter_1 = _np_reduce_scatter(np, t)
    assert np.array_equal(res_scatter_1, t)
