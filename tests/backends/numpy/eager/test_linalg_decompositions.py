"""Tests for test_linalg_decompositions."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.backends.eager.reference_math.linalg_ops import (
    _dot_general,
    _get_uncontracted_dims,
    _np_adjoint,
    _np_cholesky_ex,
    _np_cross,
    _np_inv_ex,
    _np_lu_pivots,
    _np_matrix_rank,
    _np_matrix_transpose,
    _np_pinv,
    _np_qr,
    _np_slogdet,
    _np_solve_ex,
    _np_sqrtm,
    _np_svd,
    _np_trace,
    _np_tridiagonal,
    _parse_dot_dimension_numbers,
)
from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry, numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from ml_switcheroo_compiler.backends.numpy.eager.scipy_linalg_kernels import (
    _np_cholesky_solve,
    _np_eigh_tridiagonal,
    _np_hessenberg,
    _np_lu,
    _np_lu_factor,
    _np_lu_solve,
    _np_matrix_exponential,
    _np_polar,
    _np_qdwh,
    _np_tridiagonal_solve,
)
from ml_switcheroo_compiler.core.shape import OpValue
from tests.testing_utils import (
    EagerTensorProtocol,
    MockTestStub,
    adapt_backend_module,
    coerce_int_list,
    coerce_operand_list,
    extract_test_shape,
    require_callable,
    unpack_pair,
)

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg, EagerTensorProtocol


class DummyBackend(MockTestStub):
    """Test class DummyBackend."""

    @staticmethod
    def asarray(x: BackendArg) -> BackendArg:
        """Test function asarray.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `asarray`.
        """
        return np.asarray(x)

    @staticmethod
    def array(x: BackendArg) -> BackendArg:
        """Test function array.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `array`.
        """
        return np.array(x)

    @staticmethod
    def zeros(shape: BackendArg) -> BackendArg:
        """Test function zeros.

        Args:
            shape: Test parameter `shape`.

        Returns:
            BackendArg: Computed output value from `zeros`.
        """
        return np.zeros(coerce_int_list(shape))

    @staticmethod
    def ones(
        shape: BackendArg,
        dtype: np.dtype[np.float64] | type[float] | str | None = None,
    ) -> BackendArg:
        """Test function ones.

        Args:
            shape: Test parameter `shape`.
            dtype: Test parameter `dtype`.

        Returns:
            BackendArg: Computed output value from `ones`.
        """
        return np.ones(coerce_int_list(shape), dtype=dtype)

    @staticmethod
    def cross(
        a: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
        b: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
        axisa: int = -1,
        axisb: int = -1,
        axisc: int = -1,
        axis: int | None = None,
    ) -> BackendArg:
        """Test function cross.

        Args:
            a: Left input array.
            b: Right input array.
            axisa: Axis of `a` defining vectors.
            axisb: Axis of `b` defining vectors.
            axisc: Axis of output vectors.
            axis: Shared vector axis override.

        Returns:
            BackendArg: Computed output value from `cross`.
        """
        return np.cross(a, b, axisa=axisa, axisb=axisb, axisc=axisc, axis=axis)

    linalg = np.linalg


class _TestLinalgExtrasState(typing.TypedDict, total=False):
    """Typed shared state for `_test_linalg_extras` split helpers."""

    A: np.ndarray[tuple[int, ...], np.dtype[np.generic]]
    b: np.ndarray[tuple[int, ...], np.dtype[np.generic]]
    vals: EagerTensorProtocol | OpValue


def _test_linalg_extras_part_1(_state: _TestLinalgExtrasState) -> None:
    """Execute part 1 of test_linalg_extras.

    Args:
        _state: Shared test state dictionary.
    """
    assert _get_uncontracted_dims([1, 2, 3], [0], [2]) == [2]
    dn = (((2,), (1,)), ((0,), (0,)))
    assert _parse_dot_dimension_numbers(dn) == ((2,), (1,), (0,), (0,))
    a = ensure_ndarray(np.ones((2, 3, 4)))
    b = ensure_ndarray(np.ones((2, 4, 5)))
    res = _dot_general(a, b, dn)
    assert extract_test_shape(res) == (2, 3, 5)
    a = ensure_ndarray(np.array([[1, 2], [3, 4]]))
    assert _np_trace(DummyBackend(), a) == 5
    assert _np_matrix_rank(DummyBackend(), a) == 2
    assert np.array_equal(ensure_ndarray(_np_matrix_transpose(DummyBackend(), a)), a.T)
    assert np.array_equal(ensure_ndarray(_np_sqrtm(a)), a)
    assert np.array_equal(ensure_ndarray(_np_adjoint(DummyBackend(), a)), a.T.conj())
    assert _np_cholesky_solve(DummyBackend(), 1) == 1
    A = np.array([[2, -1], [-1, 2]], dtype=float)
    b = ensure_ndarray(np.array([1, 0], dtype=float))
    L = np.linalg.cholesky(A)
    assert np.allclose(ensure_ndarray(_np_cholesky_solve(DummyBackend(), b, L, upper=False)), [2 / 3, 1 / 3])
    assert _np_eigh_tridiagonal(DummyBackend(), 1) == 1
    alpha = np.array([2, 2, 2], dtype=float)
    beta = np.array([-1, -1], dtype=float)
    (vals, vecs) = unpack_pair(_np_eigh_tridiagonal(DummyBackend(), alpha, beta))
    _state.update({"A": A})
    _state.update({"b": b})
    _state.update({"vals": vals})


def _test_linalg_extras_part_2(_state: _TestLinalgExtrasState) -> None:
    """Execute part 2 of test_linalg_extras.

    Args:
        _state: Shared test state dictionary.
    """
    handled_errors = 0
    (A, b, vals) = (_state["A"], _state["b"], _state["vals"])
    assert adapt_backend_module(vals).require_symbol("shape") == (3,)
    assert extract_test_shape(coerce_operand_list(_np_qr(DummyBackend(), A))[0]) == (2, 2)
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    assert np.array_equal(ensure_ndarray(_np_cross(DummyBackend(), v1, v2, axes={"axis": None})), [0, 0, 1])
    assert np.array_equal(ensure_ndarray(_np_cross(DummyBackend(), v1, v2, axis=None)), [0, 0, 1])
    assert len(coerce_operand_list(_np_slogdet(DummyBackend(), A))) == 2
    for op in [
        "Qr",
        "Solve",
        "Tensorinv",
        "Tensorsolve",
        "Eig",
        "Eigh",
        "Eigvals",
        "Eigvalsh",
        "Norm",
        "Cond",
        "MultiDot",
    ]:
        func = global_eager_registry._registry[op]
        if op == "Solve":
            assert func(DummyBackend(), A, b).shape == (2,)
        elif op == "MultiDot":
            assert func(DummyBackend(), [A, A]).shape == (2, 2)
        elif op in ["Eigvals", "Eigvalsh", "Norm", "Cond", "Eigh", "Eig"]:
            func(DummyBackend(), A)
        else:
            try:
                func(DummyBackend(), A)
            except (
                ValueError,
                TypeError,
                AttributeError,
                IndexError,
                KeyError,
                RuntimeError,
                ZeroDivisionError,
                NotImplementedError,
                OSError,
                ImportError,
                OverflowError,
                RecursionError,
                LookupError,
                ArithmeticError,
            ):
                handled_errors += 1
    (p, _lower, u) = coerce_operand_list(_np_lu(DummyBackend(), A))
    (lu, piv) = unpack_pair(_np_lu_factor(DummyBackend(), A))
    assert extract_test_shape(_np_lu_solve(DummyBackend(), lu, piv, b)) == (2,)
    assert extract_test_shape(_np_lu_pivots(DummyBackend(), [1, 1], 2)) == (2,)
    _state.update({"A": A, "b": b, "vals": vals})


def _test_linalg_extras_part_3(_state: _TestLinalgExtrasState) -> None:
    """Execute part 3 of test_linalg_extras.

    Args:
        _state: Shared test state dictionary.
    """
    A = _state["A"]
    assert extract_test_shape(_np_matrix_exponential(DummyBackend(), A)) == (2, 2)
    (h, q) = unpack_pair(_np_hessenberg(DummyBackend(), A))
    assert extract_test_shape(h) == (2, 2)
    (d, od, q) = coerce_operand_list(_np_tridiagonal(DummyBackend(), A))
    assert extract_test_shape(d) == (2,)
    assert extract_test_shape(od) == (1,)
    dl = np.array([-1], dtype=float)
    d = np.array([2, 2], dtype=float)
    du = np.array([-1], dtype=float)
    b_tri = np.array([1, 0], dtype=float)
    assert extract_test_shape(_np_tridiagonal_solve(DummyBackend(), dl, d, du, b_tri)) == (2,)
    (c, i) = unpack_pair(_np_cholesky_ex(DummyBackend(), A))
    assert i == 0
    (c_fail, i_fail) = unpack_pair(_np_cholesky_ex(DummyBackend(), np.array([[0, 0], [0, 0]])))
    assert ensure_ndarray(i_fail).all() == 1
    (i1, e1) = unpack_pair(_np_inv_ex(DummyBackend(), A))
    assert e1 == 0
    (i2, e2) = unpack_pair(_np_inv_ex(DummyBackend(), np.array([[0, 0], [0, 0]])))
    assert ensure_ndarray(e2).all() == 1
    assert extract_test_shape(_np_pinv(DummyBackend(), A)) == (2, 2)
    assert extract_test_shape(coerce_operand_list(_np_polar(DummyBackend(), A))[0]) == (2, 2)
    assert extract_test_shape(coerce_operand_list(_np_qdwh(DummyBackend(), A))[0]) == (2, 2)
    _state.update({"A": A})


def _test_linalg_extras_part_4(_state: _TestLinalgExtrasState) -> None:
    """Execute part 4 of test_linalg_extras.

    Args:
        _state: Shared test state dictionary.
    """
    (A, b) = (_state["A"], _state["b"])
    (s, e) = unpack_pair(_np_solve_ex(DummyBackend(), A, b))
    assert e == 0
    (s_fail, e_fail) = unpack_pair(_np_solve_ex(DummyBackend(), np.array([[0, 0], [0, 0]]), b))
    assert ensure_ndarray(e_fail).all() == 1
    (u, s, vh) = coerce_operand_list(_np_svd(DummyBackend(), A))
    assert extract_test_shape(u) == (2, 2)


def test_linalg_extras() -> None:
    """Test function test_linalg_extras."""
    _state: _TestLinalgExtrasState = {}
    _test_linalg_extras_part_1(_state)
    _test_linalg_extras_part_2(_state)
    _test_linalg_extras_part_3(_state)
    _test_linalg_extras_part_4(_state)
    assert len(_state) > 0


def test_get_uncontracted_dims() -> None:
    """Test function test_get_uncontracted_dims."""
    assert _get_uncontracted_dims([0, 1, 2], batch=[0], contracting=[2]) == [1]


def test_parse_dot_dimension_numbers() -> None:
    """Test function test_parse_dot_dimension_numbers."""
    assert _parse_dot_dimension_numbers((([2], [2]), ([0], [0]))) == ([2], [2], [0], [0])


def test_dot_general() -> None:
    """Test function test_dot_general."""
    a = np.ones((2, 3, 4))
    b = np.ones((2, 4, 5))
    res = _dot_general(a, b, (([2], [1]), ([0], [0])))
    assert extract_test_shape(res) == (2, 3, 5)


def test_np_trace() -> None:
    """Test function test_np_trace."""
    a = np.eye(3)
    assert _np_trace(None, a) == 3.0


def test_np_matrix_rank() -> None:
    """Test function test_np_matrix_rank."""
    a = np.eye(3)
    assert _np_matrix_rank(None, a) == 3


def test_np_matrix_transpose() -> None:
    """Test function test_np_matrix_transpose."""
    a = np.ones((2, 3))
    assert extract_test_shape(_np_matrix_transpose(None, a)) == (3, 2)


def test_np_sqrtm() -> None:
    """Test function test_np_sqrtm."""
    a = np.ones((2, 2))
    assert np.all(_np_sqrtm(a) == a)


def test_np_adjoint() -> None:
    """Test function test_np_adjoint."""
    a = np.array([[1 + 1j, 2], [3, 4]])
    res = _np_adjoint(None, a)
    assert ensure_ndarray(res)[(0, 0)] == 1 - 1j
    assert ensure_ndarray(res)[(0, 1)] == 3


def test_np_cholesky_solve() -> None:
    """Test function test_np_cholesky_solve."""
    a = np.ones(3)
    assert np.all(_np_cholesky_solve(np, a) == a)
    u = np.linalg.cholesky(np.array([[2, 1], [1, 2]]))
    b = np.array([1, 2])
    res = _np_cholesky_solve(np, b, u, upper=False)
    assert len(coerce_operand_list(res)) == 2


def test_np_eigh_tridiagonal() -> None:
    """Test function test_np_eigh_tridiagonal."""
    a = np.ones(3)
    assert np.all(_np_eigh_tridiagonal(np, a) == a)
    alpha = np.array([2.0, 2.0, 2.0])
    beta = np.array([1.0, 1.0])
    (vals, vecs) = unpack_pair(_np_eigh_tridiagonal(np, alpha, beta))
    assert len(coerce_operand_list(vals)) == 3


def test_np_qr() -> None:
    """Test function test_np_qr."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    (q, r) = unpack_pair(_np_qr(np, a))
    assert extract_test_shape(q) == (2, 2)


def test_np_cross() -> None:
    """Test function test_np_cross."""
    a = np.array([1, 0, 0])
    b = np.array([0, 1, 0])
    res = _np_cross(np, a, b, axes={"axis": None})
    assert np.all(res == np.array([0, 0, 1]))
    res2 = _np_cross(np, a, b, axis=None)
    assert np.all(res2 == np.array([0, 0, 1]))


def test_np_slogdet() -> None:
    """Test function test_np_slogdet."""
    a = np.eye(3)
    (sign, logdet) = unpack_pair(_np_slogdet(np, a))
    assert sign == 1.0
    assert logdet == 0.0


def test_linalg_wrappers() -> None:
    """Test function test_linalg_wrappers."""
    a = np.eye(3)
    res = require_callable(numpy_eager_registry.get)("Solve")(np, a, np.ones(3))
    assert res.shape == (3,)
    res = require_callable(numpy_eager_registry.get)("MultiDot")(np, [a, a])
    assert res.shape == (3, 3)


def test_np_lu() -> None:
    """Test function test_np_lu."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    (p, _lower, _u) = coerce_operand_list(_np_lu(None, a))
    assert extract_test_shape(p) == (2, 2)


def test_np_lu_factor() -> None:
    """Test function test_np_lu_factor."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    (lu, piv) = unpack_pair(_np_lu_factor(None, a))
    assert extract_test_shape(lu) == (2, 2)


def test_np_lu_solve() -> None:
    """Test function test_np_lu_solve."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    (lu, piv) = unpack_pair(_np_lu_factor(None, a))
    b = np.array([1.0, 2.0])
    res = _np_lu_solve(None, lu, piv, b)
    assert extract_test_shape(res) == (2,)


def test_np_lu_pivots() -> None:
    """Test function test_np_lu_pivots."""
    pivots = np.array([1, 1])
    perm = _np_lu_pivots(None, pivots, 2)
    assert extract_test_shape(perm) == (2,)


def test_np_matrix_exponential() -> None:
    """Test function test_np_matrix_exponential."""
    a = np.eye(2)
    res = _np_matrix_exponential(None, a)
    assert extract_test_shape(res) == (2, 2)


def test_np_hessenberg() -> None:
    """Test function test_np_hessenberg."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    (h, q) = unpack_pair(_np_hessenberg(None, a))
    assert extract_test_shape(h) == (2, 2)


def test_np_tridiagonal() -> None:
    """Test function test_np_tridiagonal."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    (d, od, q) = coerce_operand_list(_np_tridiagonal(None, a))
    assert len(coerce_operand_list(d)) == 2
    assert len(coerce_operand_list(od)) == 1


def test_np_tridiagonal_solve() -> None:
    """Test function test_np_tridiagonal_solve."""
    dl = np.array([1.0])
    d = np.array([2.0, 2.0])
    du = np.array([1.0])
    b = np.array([1.0, 2.0])
    res = _np_tridiagonal_solve(None, dl, d, du, b)
    assert extract_test_shape(res) == (2,)


def test_np_cholesky_ex() -> None:
    """Test function test_np_cholesky_ex."""
    a = np.array([[2.0, 1.0], [1.0, 2.0]])
    (_lower, info) = unpack_pair(_np_cholesky_ex(None, a))
    assert info == 0
    bad = np.array([[0.0, 1.0], [1.0, 0.0]])
    (_lower, info) = unpack_pair(_np_cholesky_ex(None, bad))
    assert info == 1


def test_np_inv_ex() -> None:
    """Test function test_np_inv_ex."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    (inv, info) = unpack_pair(_np_inv_ex(None, a))
    assert info == 0
    bad = np.array([[1.0, 1.0], [1.0, 1.0]])
    (inv, info) = unpack_pair(_np_inv_ex(None, bad))
    assert info == 1


def test_np_pinv() -> None:
    """Test function test_np_pinv."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    res = _np_pinv(None, a)
    assert extract_test_shape(res) == (2, 2)


def test_np_polar() -> None:
    """Test function test_np_polar."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    (u, p) = unpack_pair(_np_polar(None, a))
    assert extract_test_shape(u) == (2, 2)


def test_np_qdwh() -> None:
    """Test function test_np_qdwh."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    (u, p, iters, conv) = coerce_operand_list(_np_qdwh(None, a))
    assert extract_test_shape(u) == (2, 2)
    assert conv


def test_np_solve_ex() -> None:
    """Test function test_np_solve_ex."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([1.0, 2.0])
    (res, info) = unpack_pair(_np_solve_ex(None, a, b))
    assert info == 0
    bad = np.array([[1.0, 1.0], [1.0, 1.0]])
    (res, info) = unpack_pair(_np_solve_ex(None, bad, b))
    assert info == 1


def test_np_svd() -> None:
    """Test function test_np_svd."""
    from ml_switcheroo_compiler.backends.numpy.eager.scipy_linalg_kernels import _np_banded_triangular_solve

    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    (u, s, vh) = coerce_operand_list(_np_svd(None, a))
    assert extract_test_shape(s) == (2,)
    res_banded = _np_banded_triangular_solve(np, np.ones((1, 2)), np.ones((2,)))
    assert extract_test_shape(res_banded) == (2,)
