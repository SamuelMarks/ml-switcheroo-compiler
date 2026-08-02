import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.math_misc import _np_customlinearsolve


def test_custom_linear_solve_fallback():
    # If no solve, return args[1]
    def f(x):
        return x

    res = _np_customlinearsolve(np, f, np.array([1, 2]))
    assert np.array_equal(res, np.array([1, 2]))
