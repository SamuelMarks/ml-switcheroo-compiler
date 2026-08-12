import numpy as np

from ml_switcheroo_compiler.backends.eager.core_math_ops import _pmean, _psum
from ml_switcheroo_compiler.backends.numpy.eager.distributed import _np_all_gather, _np_all_reduce, _np_all_to_all, set_np_distributed_context
from ml_switcheroo_compiler.backends.numpy.eager.math_advanced import _np_psum


def test_math_misc_distributed():
    set_np_distributed_context(world_size=1, rank=0, addr="127.0.0.1", port=40001)

    t = np.array([1, 2])

    res = _np_all_gather(np, t, axis=0)

    res2 = _np_all_reduce(np, t, op_type="sum")

    res3 = _np_psum(np, t)

    res4 = _np_all_to_all(np, t)


def test_core_math_ops_distributed():
    t = np.array([1, 2])
    assert np.array_equal(_psum(np, t), t)
    assert np.array_equal(_pmean(np, t), t)
