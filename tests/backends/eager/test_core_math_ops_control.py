import numpy as np

from ml_switcheroo_compiler.backends.eager.core_math_ops import _associative_scan


def test_associative_scan():
    fn = lambda a, b: a + b
    elems = np.array([1, 2, 3, 4])

    # scan over axis 0
    res = _associative_scan(np, fn, elems)
    assert np.array_equal(res, np.array([1, 3, 6, 10]))

    # 2d scan over axis 1
    elems2d = np.array([[1, 2, 3], [4, 5, 6]])
    res2d = _associative_scan(np, fn, elems2d, axis=1)
    assert np.array_equal(res2d, np.array([[1, 3, 6], [4, 9, 15]]))
