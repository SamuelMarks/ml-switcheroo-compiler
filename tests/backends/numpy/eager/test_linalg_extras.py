import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.linalg_extras import (
    _dot_general,
    _get_uncontracted_dims,
    _np_adjoint,
    _np_cholesky_ex,
    _np_cholesky_solve,
    _np_cross,
    _np_eigh_tridiagonal,
    _np_hessenberg,
    _np_inv_ex,
    _np_lu,
    _np_lu_factor,
    _np_lu_pivots,
    _np_lu_solve,
    _np_matrix_exponential,
    _np_matrix_rank,
    _np_matrix_transpose,
    _np_pinv,
    _np_polar,
    _np_qdwh,
    _np_qr,
    _np_slogdet,
    _np_solve_ex,
    _np_sqrtm,
    _np_svd,
    _np_trace,
    _np_tridiagonal,
    _np_tridiagonal_solve,
    _parse_dot_dimension_numbers,
)


class DummyBackend:
    @staticmethod
    def asarray(x):
        return np.asarray(x)

    @staticmethod
    def array(x):
        return np.array(x)

    @staticmethod
    def zeros(shape):
        return np.zeros(shape)

    @staticmethod
    def ones(shape, dtype):
        return np.ones(shape, dtype=dtype)

    @staticmethod
    def cross(*args, **kwargs):
        return np.cross(*args, **kwargs)

    linalg = np.linalg


def test_linalg_extras():
    assert _get_uncontracted_dims([1, 2, 3], [0], [2]) == [2]

    dn = (((2,), (1,)), ((0,), (0,)))
    assert _parse_dot_dimension_numbers(dn) == ((2,), (1,), (0,), (0,))

    a = np.ones((2, 3, 4))
    b = np.ones((2, 4, 5))
    res = _dot_general(a, b, dn)
    assert res.shape == (2, 3, 5)

    a = np.array([[1, 2], [3, 4]])
    assert _np_trace(DummyBackend(), a) == 5
    assert _np_matrix_rank(DummyBackend(), a) == 2
    assert np.array_equal(_np_matrix_transpose(DummyBackend(), a), a.T)
    assert np.array_equal(_np_sqrtm(a), a)
    assert np.array_equal(_np_adjoint(DummyBackend(), a), a.T.conj())

    # CholeskySolve
    assert _np_cholesky_solve(DummyBackend(), 1) == 1
    A = np.array([[2, -1], [-1, 2]], dtype=float)
    b = np.array([1, 0], dtype=float)
    L = np.linalg.cholesky(A)
    assert np.allclose(_np_cholesky_solve(DummyBackend(), b, L, upper=False), [2 / 3, 1 / 3])

    # EighTridiagonal
    assert _np_eigh_tridiagonal(DummyBackend(), 1) == 1
    alpha = np.array([2, 2, 2], dtype=float)
    beta = np.array([-1, -1], dtype=float)
    vals, vecs = _np_eigh_tridiagonal(DummyBackend(), alpha, beta)
    assert vals.shape == (3,)

    # Qr
    assert _np_qr(DummyBackend(), A)[0].shape == (2, 2)

    # Cross
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    assert np.array_equal(_np_cross(DummyBackend(), v1, v2, axes={"axis": None}), [0, 0, 1])
    assert np.array_equal(_np_cross(DummyBackend(), v1, v2, axis=None), [0, 0, 1])

    # Slogdet
    assert len(_np_slogdet(DummyBackend(), A)) == 2

    # Op wrappers
    for op in ["Qr", "Solve", "Tensorinv", "Tensorsolve", "Eig", "Eigh", "Eigvals", "Eigvalsh", "Norm", "Cond", "MultiDot"]:
        func = numpy_eager_registry._registry[op]
        if op == "Solve":
            assert func(DummyBackend(), A, b).shape == (2,)
        elif op == "MultiDot":
            assert func(DummyBackend(), [A, A]).shape == (2, 2)
        elif op in ["Eigvals", "Eigvalsh", "Norm", "Cond", "Eigh", "Eig"]:
            res = func(DummyBackend(), A)
        else:
            try:
                func(DummyBackend(), A)
            except:
                pass

    # LU
    p, l, u = _np_lu(DummyBackend(), A)
    lu, piv = _np_lu_factor(DummyBackend(), A)
    assert _np_lu_solve(DummyBackend(), lu, piv, b).shape == (2,)
    assert _np_lu_pivots(DummyBackend(), [1, 1], 2).shape == (2,)

    # Expm
    assert _np_matrix_exponential(DummyBackend(), A).shape == (2, 2)

    # Hessenberg
    h, q = _np_hessenberg(DummyBackend(), A)
    assert h.shape == (2, 2)

    # Tridiagonal
    d, od, q = _np_tridiagonal(DummyBackend(), A)
    assert d.shape == (2,)
    assert od.shape == (1,)

    # TridiagonalSolve
    dl = np.array([-1], dtype=float)
    d = np.array([2, 2], dtype=float)
    du = np.array([-1], dtype=float)
    b_tri = np.array([1, 0], dtype=float)
    assert _np_tridiagonal_solve(DummyBackend(), dl, d, du, b_tri).shape == (2,)

    # CholeskyEx
    c, i = _np_cholesky_ex(DummyBackend(), A)
    assert i == 0
    c_fail, i_fail = _np_cholesky_ex(DummyBackend(), np.array([[0, 0], [0, 0]]))
    assert i_fail.all() == 1

    # InvEx
    i1, e1 = _np_inv_ex(DummyBackend(), A)
    assert e1 == 0
    i2, e2 = _np_inv_ex(DummyBackend(), np.array([[0, 0], [0, 0]]))
    assert e2.all() == 1

    # Pinv
    assert _np_pinv(DummyBackend(), A).shape == (2, 2)

    # Polar / Qdwh
    assert _np_polar(DummyBackend(), A)[0].shape == (2, 2)
    assert _np_qdwh(DummyBackend(), A)[0].shape == (2, 2)

    # SolveEx
    s, e = _np_solve_ex(DummyBackend(), A, b)
    assert e == 0
    s_fail, e_fail = _np_solve_ex(DummyBackend(), np.array([[0, 0], [0, 0]]), b)
    assert e_fail.all() == 1

    # Svd
    u, s, vh = _np_svd(DummyBackend(), A)
    assert u.shape == (2, 2)
