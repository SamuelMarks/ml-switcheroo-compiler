"""Tests for test_linalg_decompositions."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.linalg_advanced import (
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
from tests.testing_utils import as_callable, as_mod, as_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


class DummyBackend:
    """Test class DummyBackend."""

    @staticmethod
    def asarray(x: EagerValue) -> EagerValue:
        """Test function asarray.

        Args:
            x: Test parameter `x`.
        """
        return np.asarray(x)

    @staticmethod
    def array(x: EagerValue) -> EagerValue:
        """Test function array.

        Args:
            x: Test parameter `x`.
        """
        return np.array(x)

    @staticmethod
    def zeros(shape: EagerValue) -> EagerValue:
        """Test function zeros.

        Args:
            shape: Test parameter `shape`.
        """
        return as_callable(np.zeros)(shape)

    @staticmethod
    def ones(shape: EagerValue, dtype: EagerValue) -> EagerValue:
        """Test function ones.

        Args:
            shape: Test parameter `shape`.
            dtype: Test parameter `dtype`.
        """
        return as_callable(np.ones)(shape, dtype=dtype)

    @staticmethod
    def cross(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function cross.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return as_callable(np.cross)(*args, **kwargs)

    linalg = np.linalg


def _test_linalg_extras_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_linalg_extras.

    Args:
        _state: Shared test state dictionary.
    """
    assert _get_uncontracted_dims([1, 2, 3], [0], [2]) == [2]
    dn = (((2,), (1,)), ((0,), (0,)))
    assert as_callable(_parse_dot_dimension_numbers)(dn) == ((2,), (1,), (0,), (0,))
    a = np.ones((2, 3, 4))
    b = np.ones((2, 4, 5))
    res = _dot_general(a, b, dn)
    assert as_tensor(getattr(res, "shape")) == (2, 3, 5)
    a = np.array([[1, 2], [3, 4]])
    assert _np_trace(as_mod(DummyBackend()), a) == 5
    assert _np_matrix_rank(as_mod(DummyBackend()), a) == 2
    assert as_callable(np.array_equal)(as_callable(_np_matrix_transpose)(as_callable(DummyBackend)(), a), a.T)
    assert as_callable(np.array_equal)(as_callable(_np_sqrtm)(a), a)
    assert as_callable(np.array_equal)(as_callable(_np_adjoint)(as_callable(DummyBackend)(), a), as_callable(a.T.conj)())
    assert _np_cholesky_solve(as_mod(DummyBackend()), 1) == 1
    A = np.array([[2, -1], [-1, 2]], dtype=float)
    b = np.array([1, 0], dtype=float)
    L = np.linalg.cholesky(A)
    assert as_callable(np.allclose)(as_callable(_np_cholesky_solve)(as_callable(DummyBackend)(), b, L, upper=False), [2 / 3, 1 / 3])
    assert _np_eigh_tridiagonal(as_mod(DummyBackend()), 1) == 1
    alpha = np.array([2, 2, 2], dtype=float)
    beta = np.array([-1, -1], dtype=float)
    (vals, vecs) = as_callable(_np_eigh_tridiagonal)(as_callable(DummyBackend)(), alpha, beta)
    getattr(_state, "update")({"A": A, "L": L, "a": a, "alpha": alpha})
    getattr(_state, "update")({"b": b, "beta": beta, "dn": dn, "res": res})
    getattr(_state, "update")({"vals": vals, "vecs": vecs})
    _state.update(locals())


def _test_linalg_extras_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_linalg_extras.

    Args:
        _state: Shared test state dictionary.
    """
    (A, b, vals) = (_state["A"], _state["b"], _state["vals"])
    assert vals.shape == (3,)
    assert as_tensor(getattr(as_tensor(getattr(_np_qr(as_mod(DummyBackend()), A), "__getitem__")(0)), "shape")) == (2, 2)
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    assert as_callable(np.array_equal)(as_callable(_np_cross)(as_callable(DummyBackend)(), v1, v2, axes={"axis": None}), [0, 0, 1])
    assert as_callable(np.array_equal)(as_callable(_np_cross)(as_callable(DummyBackend)(), v1, v2, axis=None), [0, 0, 1])
    assert as_callable(len)(as_callable(_np_slogdet)(as_callable(DummyBackend)(), A)) == 2
    for op in ["Qr", "Solve", "Tensorinv", "Tensorsolve", "Eig", "Eigh", "Eigvals", "Eigvalsh", "Norm", "Cond", "MultiDot"]:
        func = numpy_eager_registry._registry[op]
        if op == "Solve":
            assert as_tensor(getattr(func(DummyBackend(), A, b), "shape")) == (2,)
        elif op == "MultiDot":
            assert as_tensor(getattr(func(DummyBackend(), [A, A]), "shape")) == (2, 2)
        elif op in ["Eigvals", "Eigvalsh", "Norm", "Cond", "Eigh", "Eig"]:
            res = func(DummyBackend(), A)
        else:
            try:
                func(DummyBackend(), A)
            except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
                _caught_err = True
    (p, _lower, u) = as_callable(_np_lu)(as_callable(DummyBackend)(), A)
    (lu, piv) = as_callable(_np_lu_factor)(as_callable(DummyBackend)(), A)
    assert as_tensor(getattr(_np_lu_solve(as_mod(DummyBackend()), lu, piv, b), "shape")) == (2,)
    assert as_tensor(getattr(_np_lu_pivots(as_mod(DummyBackend()), [1, 1], 2), "shape")) == (2,)
    getattr(_state, "update")({"_lower": _lower, "lu": lu, "p": p, "piv": piv})
    getattr(_state, "update")({"u": u, "v1": v1, "v2": v2})
    _state.update(locals())


def _test_linalg_extras_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_linalg_extras.

    Args:
        _state: Shared test state dictionary.
    """
    A = _state["A"]
    assert as_tensor(getattr(_np_matrix_exponential(as_mod(DummyBackend()), A), "shape")) == (2, 2)
    (h, q) = as_callable(_np_hessenberg)(as_callable(DummyBackend)(), A)
    assert as_tensor(getattr(h, "shape")) == (2, 2)
    (d, od, q) = as_callable(_np_tridiagonal)(as_callable(DummyBackend)(), A)
    assert as_tensor(getattr(d, "shape")) == (2,)
    assert as_tensor(getattr(od, "shape")) == (1,)
    dl = np.array([-1], dtype=float)
    d = as_tensor(np.array([2, 2], dtype=float))
    du = np.array([-1], dtype=float)
    b_tri = np.array([1, 0], dtype=float)
    assert as_tensor(getattr(_np_tridiagonal_solve(as_mod(DummyBackend()), dl, d, du, b_tri), "shape")) == (2,)
    (c, i) = as_callable(_np_cholesky_ex)(as_callable(DummyBackend)(), A)
    assert i == 0
    (c_fail, i_fail) = as_callable(_np_cholesky_ex)(as_callable(DummyBackend)(), as_callable(np.array)([[0, 0], [0, 0]]))
    assert as_tensor(getattr(i_fail, "all"))() == 1
    (i1, e1) = as_callable(_np_inv_ex)(as_callable(DummyBackend)(), A)
    assert e1 == 0
    (i2, e2) = as_callable(_np_inv_ex)(as_callable(DummyBackend)(), as_callable(np.array)([[0, 0], [0, 0]]))
    assert as_tensor(getattr(e2, "all"))() == 1
    assert as_tensor(getattr(_np_pinv(as_mod(DummyBackend()), A), "shape")) == (2, 2)
    assert as_tensor(getattr(as_tensor(getattr(_np_polar(as_mod(DummyBackend()), A), "__getitem__")(0)), "shape")) == (2, 2)
    assert as_tensor(getattr(as_tensor(getattr(_np_qdwh(as_mod(DummyBackend()), A), "__getitem__")(0)), "shape")) == (2, 2)
    getattr(_state, "update")({"b_tri": b_tri, "c": c, "c_fail": c_fail, "d": d})
    getattr(_state, "update")({"dl": dl, "du": du, "e1": e1, "e2": e2})
    getattr(_state, "update")({"h": h, "i": i, "i1": i1, "i2": i2})
    getattr(_state, "update")({"i_fail": i_fail, "od": od, "q": q})
    _state.update(locals())


def _test_linalg_extras_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_linalg_extras.

    Args:
        _state: Shared test state dictionary.
    """
    (A, b) = (_state["A"], _state["b"])
    (s, e) = as_callable(_np_solve_ex)(as_callable(DummyBackend)(), A, b)
    assert e == 0
    (s_fail, e_fail) = as_callable(_np_solve_ex)(as_callable(DummyBackend)(), as_callable(np.array)([[0, 0], [0, 0]]), b)
    assert as_tensor(getattr(e_fail, "all"))() == 1
    (u, s, vh) = as_callable(_np_svd)(as_callable(DummyBackend)(), A)
    assert as_tensor(getattr(u, "shape")) == (2, 2)
    getattr(_state, "update")({"e": e, "e_fail": e_fail, "s": s, "s_fail": s_fail})
    getattr(_state, "update")({"u": u, "vh": vh})


def test_linalg_extras() -> None:
    """Test function test_linalg_extras."""
    _state: dict[str, EagerTensorProtocol] = {}
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
    assert as_tensor(getattr(res, "shape")) == (2, 3, 5)


def test_np_trace() -> None:
    """Test function test_np_trace."""
    a = np.eye(3)
    assert _np_trace(as_mod(None), a) == 3.0


def test_np_matrix_rank() -> None:
    """Test function test_np_matrix_rank."""
    a = np.eye(3)
    assert _np_matrix_rank(as_mod(None), a) == 3


def test_np_matrix_transpose() -> None:
    """Test function test_np_matrix_transpose."""
    a = np.ones((2, 3))
    assert as_tensor(getattr(_np_matrix_transpose(as_mod(None), a), "shape")) == (3, 2)


def test_np_sqrtm() -> None:
    """Test function test_np_sqrtm."""
    a = np.ones((2, 2))
    assert np.all(_np_sqrtm(a) == a)


def test_np_adjoint() -> None:
    """Test function test_np_adjoint."""
    a = np.array([[1 + 1j, 2], [3, 4]])
    res = _np_adjoint(as_mod(None), a)
    assert as_tensor(getattr(res, "__getitem__")((0, 0))) == 1 - 1j
    assert as_tensor(getattr(res, "__getitem__")((0, 1))) == 3


def test_np_cholesky_solve() -> None:
    """Test function test_np_cholesky_solve."""
    a = np.ones(3)
    assert np.all(_np_cholesky_solve(np, a) == a)
    u = np.linalg.cholesky(np.array([[2, 1], [1, 2]]))
    b = np.array([1, 2])
    res = _np_cholesky_solve(np, b, u, upper=False)
    assert as_callable(len)(res) == 2


def test_np_eigh_tridiagonal() -> None:
    """Test function test_np_eigh_tridiagonal."""
    a = np.ones(3)
    assert np.all(_np_eigh_tridiagonal(np, a) == a)
    alpha = np.array([2.0, 2.0, 2.0])
    beta = np.array([1.0, 1.0])
    (vals, vecs) = as_callable(_np_eigh_tridiagonal)(np, alpha, beta)
    assert as_callable(len)(vals) == 3


def test_np_qr() -> None:
    """Test function test_np_qr."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    (q, r) = as_callable(_np_qr)(np, a)
    assert as_tensor(getattr(q, "shape")) == (2, 2)


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
    (sign, logdet) = as_callable(_np_slogdet)(np, a)
    assert sign == 1.0
    assert logdet == 0.0


def test_linalg_wrappers() -> None:
    """Test function test_linalg_wrappers."""
    a = np.eye(3)
    res = as_callable(numpy_eager_registry.get)("Solve")(np, a, as_callable(np.ones)(3))
    assert as_tensor(getattr(res, "shape")) == (3,)
    res = as_callable(numpy_eager_registry.get)("MultiDot")(np, [a, a])
    assert as_tensor(getattr(res, "shape")) == (3, 3)


def test_np_lu() -> None:
    """Test function test_np_lu."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    (p, _lower, _u) = as_callable(_np_lu)(None, a)
    assert as_tensor(getattr(p, "shape")) == (2, 2)


def test_np_lu_factor() -> None:
    """Test function test_np_lu_factor."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    (lu, piv) = as_callable(_np_lu_factor)(None, a)
    assert as_tensor(getattr(lu, "shape")) == (2, 2)


def test_np_lu_solve() -> None:
    """Test function test_np_lu_solve."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    (lu, piv) = as_callable(_np_lu_factor)(None, a)
    b = np.array([1.0, 2.0])
    res = _np_lu_solve(as_mod(None), lu, piv, b)
    assert as_tensor(getattr(res, "shape")) == (2,)


def test_np_lu_pivots() -> None:
    """Test function test_np_lu_pivots."""
    pivots = np.array([1, 1])
    perm = _np_lu_pivots(as_mod(None), pivots, 2)
    assert as_tensor(getattr(perm, "shape")) == (2,)


def test_np_matrix_exponential() -> None:
    """Test function test_np_matrix_exponential."""
    a = np.eye(2)
    res = _np_matrix_exponential(as_mod(None), a)
    assert as_tensor(getattr(res, "shape")) == (2, 2)


def test_np_hessenberg() -> None:
    """Test function test_np_hessenberg."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    (h, q) = as_callable(_np_hessenberg)(None, a)
    assert as_tensor(getattr(h, "shape")) == (2, 2)


def test_np_tridiagonal() -> None:
    """Test function test_np_tridiagonal."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    (d, od, q) = as_callable(_np_tridiagonal)(None, a)
    assert as_callable(len)(d) == 2
    assert as_callable(len)(od) == 1


def test_np_tridiagonal_solve() -> None:
    """Test function test_np_tridiagonal_solve."""
    dl = np.array([1.0])
    d = np.array([2.0, 2.0])
    du = np.array([1.0])
    b = np.array([1.0, 2.0])
    res = _np_tridiagonal_solve(as_mod(None), dl, d, du, b)
    assert as_tensor(getattr(res, "shape")) == (2,)


def test_np_cholesky_ex() -> None:
    """Test function test_np_cholesky_ex."""
    a = np.array([[2.0, 1.0], [1.0, 2.0]])
    (_lower, info) = as_callable(_np_cholesky_ex)(None, a)
    assert info == 0
    bad = np.array([[0.0, 1.0], [1.0, 0.0]])
    (_lower, info) = as_callable(_np_cholesky_ex)(None, bad)
    assert info == 1


def test_np_inv_ex() -> None:
    """Test function test_np_inv_ex."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    (inv, info) = as_callable(_np_inv_ex)(None, a)
    assert info == 0
    bad = np.array([[1.0, 1.0], [1.0, 1.0]])
    (inv, info) = as_callable(_np_inv_ex)(None, bad)
    assert info == 1


def test_np_pinv() -> None:
    """Test function test_np_pinv."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    res = _np_pinv(as_mod(None), a)
    assert as_tensor(getattr(res, "shape")) == (2, 2)


def test_np_polar() -> None:
    """Test function test_np_polar."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    (u, p) = as_callable(_np_polar)(None, a)
    assert as_tensor(getattr(u, "shape")) == (2, 2)


def test_np_qdwh() -> None:
    """Test function test_np_qdwh."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    (u, p, iters, conv) = as_callable(_np_qdwh)(None, a)
    assert as_tensor(getattr(u, "shape")) == (2, 2)
    assert conv


def test_np_solve_ex() -> None:
    """Test function test_np_solve_ex."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([1.0, 2.0])
    (res, info) = as_callable(_np_solve_ex)(None, a, b)
    assert info == 0
    bad = np.array([[1.0, 1.0], [1.0, 1.0]])
    (res, info) = as_callable(_np_solve_ex)(None, bad, b)
    assert info == 1


def test_np_svd() -> None:
    """Test function test_np_svd."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    (u, s, vh) = as_callable(_np_svd)(None, a)
    assert as_tensor(getattr(s, "shape")) == (2,)
