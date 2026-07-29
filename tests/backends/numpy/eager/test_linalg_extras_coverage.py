"""Test numpy eager linalg_extras coverage."""

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


def test_get_uncontracted_dims():
    assert _get_uncontracted_dims([0, 1, 2], batch=[0], contracting=[2]) == [1]


def test_parse_dot_dimension_numbers():
    assert _parse_dot_dimension_numbers((([2], [2]), ([0], [0]))) == ([2], [2], [0], [0])


def test_dot_general():
    a = np.ones((2, 3, 4))
    b = np.ones((2, 4, 5))
    res = _dot_general(a, b, (([2], [1]), ([0], [0])))
    assert res.shape == (2, 3, 5)


def test_np_trace():
    a = np.eye(3)
    assert _np_trace(None, a) == 3.0


def test_np_matrix_rank():
    a = np.eye(3)
    assert _np_matrix_rank(None, a) == 3


def test_np_matrix_transpose():
    a = np.ones((2, 3))
    assert _np_matrix_transpose(None, a).shape == (3, 2)


def test_np_sqrtm():
    a = np.ones((2, 2))
    assert np.all(_np_sqrtm(a) == a)


def test_np_adjoint():
    a = np.array([[1 + 1j, 2], [3, 4]])
    res = _np_adjoint(None, a)
    assert res[0, 0] == 1 - 1j
    assert res[0, 1] == 3


def test_np_cholesky_solve():
    a = np.ones(3)
    assert np.all(_np_cholesky_solve(np, a) == a)  # len(args) < 2
    u = np.linalg.cholesky(np.array([[2, 1], [1, 2]]))
    b = np.array([1, 2])
    res = _np_cholesky_solve(np, b, u, upper=False)
    assert len(res) == 2


def test_np_eigh_tridiagonal():
    a = np.ones(3)
    assert np.all(_np_eigh_tridiagonal(np, a) == a)  # len(args) < 2
    alpha = np.array([2.0, 2.0, 2.0])
    beta = np.array([1.0, 1.0])
    vals, vecs = _np_eigh_tridiagonal(np, alpha, beta)
    assert len(vals) == 3


def test_np_qr():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    q, r = _np_qr(np, a)
    assert q.shape == (2, 2)


def test_np_cross():
    a = np.array([1, 0, 0])
    b = np.array([0, 1, 0])
    res = _np_cross(np, a, b, axes={"axis": None})
    assert np.all(res == np.array([0, 0, 1]))
    res2 = _np_cross(np, a, b, axis=None)
    assert np.all(res2 == np.array([0, 0, 1]))


def test_np_slogdet():
    a = np.eye(3)
    sign, logdet = _np_slogdet(np, a)
    assert sign == 1.0
    assert logdet == 0.0


def test_linalg_wrappers():
    a = np.eye(3)
    res = numpy_eager_registry.get("Solve")(np, a, np.ones(3))
    assert res.shape == (3,)
    res = numpy_eager_registry.get("MultiDot")(np, [a, a])
    assert res.shape == (3, 3)


def test_np_lu():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    p, l, u = _np_lu(None, a)
    assert p.shape == (2, 2)


def test_np_lu_factor():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    lu, piv = _np_lu_factor(None, a)
    assert lu.shape == (2, 2)


def test_np_lu_solve():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    lu, piv = _np_lu_factor(None, a)
    b = np.array([1.0, 2.0])
    res = _np_lu_solve(None, lu, piv, b)
    assert res.shape == (2,)


def test_np_lu_pivots():
    pivots = np.array([1, 1])
    perm = _np_lu_pivots(None, pivots, 2)
    assert perm.shape == (2,)


def test_np_matrix_exponential():
    a = np.eye(2)
    res = _np_matrix_exponential(None, a)
    assert res.shape == (2, 2)


def test_np_hessenberg():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    h, q = _np_hessenberg(None, a)
    assert h.shape == (2, 2)


def test_np_tridiagonal():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    d, od, q = _np_tridiagonal(None, a)
    assert len(d) == 2
    assert len(od) == 1


def test_np_tridiagonal_solve():
    dl = np.array([1.0])
    d = np.array([2.0, 2.0])
    du = np.array([1.0])
    b = np.array([1.0, 2.0])
    res = _np_tridiagonal_solve(None, dl, d, du, b)
    assert res.shape == (2,)


def test_np_cholesky_ex():
    a = np.array([[2.0, 1.0], [1.0, 2.0]])
    l, info = _np_cholesky_ex(None, a)
    assert info == 0
    bad = np.array([[0.0, 1.0], [1.0, 0.0]])
    l, info = _np_cholesky_ex(None, bad)
    assert info == 1


def test_np_inv_ex():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    inv, info = _np_inv_ex(None, a)
    assert info == 0
    bad = np.array([[1.0, 1.0], [1.0, 1.0]])
    inv, info = _np_inv_ex(None, bad)
    assert info == 1


def test_np_pinv():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    res = _np_pinv(None, a)
    assert res.shape == (2, 2)


def test_np_polar():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    u, p = _np_polar(None, a)
    assert u.shape == (2, 2)


def test_np_qdwh():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    u, p, iters, conv = _np_qdwh(None, a)
    assert u.shape == (2, 2)
    assert conv


def test_np_solve_ex():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([1.0, 2.0])
    res, info = _np_solve_ex(None, a, b)
    assert info == 0
    bad = np.array([[1.0, 1.0], [1.0, 1.0]])
    res, info = _np_solve_ex(None, bad, b)
    assert info == 1


def test_np_svd():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    u, s, vh = _np_svd(None, a)
    assert s.shape == (2,)
