import numpy as np

from ml_switcheroo_compiler.backends.eager.core_math_ops import _pmean, _psum
from ml_switcheroo_compiler.backends.numpy.eager.distributed import set_mock_distributed_context
from ml_switcheroo_compiler.backends.numpy.eager.math_advanced import _np_all_gather, _np_all_reduce, _np_all_to_all, _np_psum


def test_math_misc_distributed():
    set_mock_distributed_context(world_size=2, rank=1)

    t = np.array([1, 2])

    # math_misc AllGather
    res = _np_all_gather(np, t)
    assert np.array_equal(res, np.array([[1, 2], [1, 2]]))

    # math_misc AllReduce
    res2 = _np_all_reduce(np, t)
    assert np.array_equal(res2, np.array([2, 4]))

    # math_misc Psum
    res3 = _np_psum(np, t)
    assert np.array_equal(res3, np.array([2, 4]))

    # math_misc AllToAll
    res4 = _np_all_to_all(np, t)
    assert np.array_equal(res4, t)

    # world size 1 fallback
    set_mock_distributed_context(world_size=1, rank=0)
    assert np.array_equal(_np_all_gather(np, t), np.array([[1, 2]]))
    assert np.array_equal(_np_all_reduce(np, t), t)
    assert np.array_equal(_np_psum(np, t), t)


def test_core_math_ops_distributed():
    t = np.array([1, 2])
    assert np.array_equal(_psum(np, t), t)
    assert np.array_equal(_pmean(np, t), t)
