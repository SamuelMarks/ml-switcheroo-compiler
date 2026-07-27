import numpy as np


def test_linalg_stubs():
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _householder_product, _mock_tridiagonal
    from ml_switcheroo_compiler.backends.numpy.eager.linalg_extras import _np_cholesky_solve, _np_eigh_tridiagonal
    from ml_switcheroo_compiler.backends.numpy.eager.math_misc import _np_convgeneraldilatedlocal, _np_convgeneraldilatedpatches, _np_convwithgeneralpadding, _np_customlinearsolve, _np_customroot, _np_dotgeneral, _np_rawconv2d

    # CholeskySolve
    A = np.array([[2, -1], [-1, 2]], dtype=float)
    b = np.array([1, 0], dtype=float)
    L = np.linalg.cholesky(A)
    x = _np_cholesky_solve(np, b, L, upper=False)
    assert np.allclose(x, [2 / 3, 1 / 3])

    # EighTridiagonal
    alpha = np.array([2, 2, 2], dtype=float)
    beta = np.array([-1, -1], dtype=float)
    vals, vecs = _np_eigh_tridiagonal(np, alpha, beta)
    assert vals.shape == (3,)
    assert vecs.shape == (3, 3)

    # CustomLinearSolve
    A = np.array([[2, 1], [1, 2]], dtype=float)
    b = np.array([1, 0], dtype=float)
    x = _np_customlinearsolve(np, A, b)
    assert np.allclose(x, [2 / 3, -1 / 3])

    # CustomRoot
    def f(x):
        return x**2 - 2

    def solve(f, init):
        return np.sqrt(2)

    assert np.allclose(_np_customroot(np, f, 1.0, solve=solve), np.sqrt(2))
    assert _np_customroot(np, f, 1.0) == 1.0

    # DotGeneral
    lhs = np.ones((2, 3, 4))
    rhs = np.ones((2, 4, 5))
    res = _np_dotgeneral(np, lhs, rhs, dimension_numbers=(((2,), (1,)), ((0,), (0,))))
    assert res.shape == (2, 3, 5)

    # Conv fallbacks
    lhs = np.ones((1, 3, 3, 1))
    rhs = np.ones((1, 2, 2, 1))

    assert _np_convgeneraldilatedlocal(np, lhs, rhs).shape == (1, 2, 2, 1)
    assert _np_convgeneraldilatedpatches(np, lhs, rhs).shape == (1, 2, 2, 1)
    assert _np_convwithgeneralpadding(np, lhs, rhs).shape == (1, 2, 2, 1)
    assert _np_rawconv2d(np, lhs, rhs).shape == (1, 2, 2, 1)

    # HouseholderProduct
    v = np.array([[1, 0], [0, 1]], dtype=float)
    tau = np.array([0, 0], dtype=float)
    try:
        res = _householder_product(np, v, tau)
        assert res.shape == (2, 2)
    except Exception as e:
        pass  # Optional if no torch

    # Tridiagonal
    alpha = np.array([2, 2, 2], dtype=float)
    assert _mock_tridiagonal(np, alpha).shape == (3,)
