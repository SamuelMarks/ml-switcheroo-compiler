"""Tests for numpy eager linalg extras ops."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.linalg_advanced import (
    _build_einsum_equation,
    _dot_general,
    _get_uncontracted_dims,
    _np_adjoint,
    _np_cholesky_solve,
    _np_cross,
    _np_eigh_tridiagonal,
    _np_matrix_rank,
    _np_matrix_transpose,
    _np_qr,
    _np_slogdet,
    _np_sqrtm,
    _np_trace,
    _parse_dot_dimension_numbers,
)

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol


def test_get_uncontracted_dims() -> None:
    """Test get uncontracted dims."""
    dims = [1, 2, 3, 4]
    batch = [0]
    contracting = [2]
    res = _get_uncontracted_dims(dims, batch, contracting)
    assert res == [2, 4]


def test_parse_dot_dimension_numbers() -> None:
    """Test parse dot dimension numbers."""
    dimension_numbers = (((1,), (0,)), ((2,), (3,)))
    res = cast("EagerCallableProtocol", _parse_dot_dimension_numbers)(dimension_numbers)
    assert res == ((1,), (0,), (2,), (3,))


def test_build_einsum_equation() -> None:
    """Test build einsum equation."""
    a_ndim = 2
    b_ndim = 2
    dimension_numbers = (((1,), (0,)), ((), ()))
    a_dims, b_dims, out_dims = _build_einsum_equation(a_ndim, b_ndim, dimension_numbers)
    assert a_dims == [0, 1]
    assert b_dims == [1, 3]  # [2, 3] modified at b_contracting[i] (0) to a_dims[a_c] (a_dims[1] -> 1)
    assert out_dims == [0, 3]


def test_dot_general() -> None:
    """Test dot general."""
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    dimension_numbers = (((1,), (0,)), ((), ()))
    res = _dot_general(a, b, dimension_numbers)
    expected = np.dot(a, b)
    cast("EagerCallableProtocol", np.testing.assert_allclose)(res, expected)


def test_trace() -> None:
    """Test trace."""
    a = np.array([[1, 2], [3, 4]])
    res = _np_trace(np, a)
    assert res == 5


def test_matrix_rank() -> None:
    """Test matrix rank."""
    a = np.array([[1, 2], [3, 4]])
    res = _np_matrix_rank(np, a)
    assert res == 2


def test_matrix_transpose() -> None:
    """Test matrix transpose."""
    a = np.array([[1, 2], [3, 4]])
    res = _np_matrix_transpose(np, a)
    expected = a.T
    cast("EagerCallableProtocol", np.testing.assert_allclose)(res, expected)


def test_sqrtm() -> None:
    """Test sqrtm."""
    a = np.array([1, 2])
    res = _np_sqrtm(a)
    assert cast("EagerCallableProtocol", np.array_equal)(res, a)


def test_adjoint() -> None:
    """Test adjoint."""
    a = np.array([[1 + 1j, 2], [3, 4]])
    res = _np_adjoint(np, a)
    expected = np.conj(a.T)
    cast("EagerCallableProtocol", np.testing.assert_allclose)(res, expected)


def test_cholesky_solve() -> None:
    """Test cholesky solve."""
    a = np.array([1, 2])
    res = _np_cholesky_solve(np, a)
    assert cast("EagerCallableProtocol", np.array_equal)(res, a)


def test_eigh_tridiagonal() -> None:
    """Test eigh tridiagonal."""
    a = np.array([1, 2])
    res = _np_eigh_tridiagonal(np, a)
    assert cast("EagerCallableProtocol", np.array_equal)(res, a)


def test_qr() -> None:
    """Test qr."""
    a = np.array([[1, 2], [3, 4]])
    res = _np_qr(np, a)
    expected = np.linalg.qr(a)
    cast("EagerCallableProtocol", np.testing.assert_allclose)(cast("EagerTensorProtocol", getattr(res, "__getitem__")(0)), cast("EagerTensorProtocol", getattr(expected, "__getitem__")(0)))
    cast("EagerCallableProtocol", np.testing.assert_allclose)(cast("EagerTensorProtocol", getattr(res, "__getitem__")(1)), cast("EagerTensorProtocol", getattr(expected, "__getitem__")(1)))


def test_cross() -> None:
    """Test cross."""
    a = np.array([1, 0, 0])
    b = np.array([0, 1, 0])
    res1 = _np_cross(np, a, b)
    expected = np.cross(a, b)
    cast("EagerCallableProtocol", np.testing.assert_allclose)(res1, expected)

    res2 = _np_cross(np, a, b, axes={"axis": None})
    cast("EagerCallableProtocol", np.testing.assert_allclose)(res2, expected)

    res3 = _np_cross(np, a, b, axis=None)
    cast("EagerCallableProtocol", np.testing.assert_allclose)(res3, expected)


def test_slogdet() -> None:
    """Test slogdet."""
    a = np.array([[1, 2], [3, 4]])
    res = _np_slogdet(np, a)
    expected = np.linalg.slogdet(a)
    cast("EagerCallableProtocol", np.testing.assert_allclose)(cast("EagerTensorProtocol", getattr(res, "__getitem__")(0)), cast("EagerTensorProtocol", getattr(expected, "__getitem__")(0)))
    cast("EagerCallableProtocol", np.testing.assert_allclose)(cast("EagerTensorProtocol", getattr(res, "__getitem__")(1)), cast("EagerTensorProtocol", getattr(expected, "__getitem__")(1)))


def test_dynamically_registered_ops() -> None:
    """Test dynamically registered ops."""
    from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry

    a = np.array([[1, 2], [3, 4]])

    # test solve
    solve_fn = numpy_eager_registry.get("Solve")
    b = np.array([1, 2])
    res_solve = cast("EagerCallableProtocol", solve_fn)(np, a, b)
    cast("EagerCallableProtocol", np.testing.assert_allclose)(res_solve, cast("EagerCallableProtocol", np.linalg.solve)(a, b))

    # test eig
    eig_fn = numpy_eager_registry.get("Eig")
    res_eig = cast("EagerCallableProtocol", eig_fn)(np, a)
    cast("EagerCallableProtocol", np.testing.assert_allclose)(cast("EagerTensorProtocol", getattr(res_eig, "__getitem__")(0)), cast("EagerTensorProtocol", getattr(cast("EagerCallableProtocol", np.linalg.eig)(a), "__getitem__")(0)))

    # test norm
    norm_fn = numpy_eager_registry.get("Norm")
    res_norm = cast("EagerCallableProtocol", norm_fn)(np, a)
    cast("EagerCallableProtocol", np.testing.assert_allclose)(res_norm, cast("EagerCallableProtocol", np.linalg.norm)(a))
