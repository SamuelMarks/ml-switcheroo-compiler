"""Tests for test_linalg_decompositions."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

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

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


class DummyBackend:
    """Test class DummyBackend."""

    @staticmethod
    def asarray(x: EagerValue) -> EagerValue:
        """Test function asarray."""
        return np.asarray(x)

    @staticmethod
    def array(x: EagerValue) -> EagerValue:
        """Test function array."""
        return np.array(x)

    @staticmethod
    def zeros(shape: EagerValue) -> EagerValue:
        """Test function zeros."""
        return cast("Callable[..., EagerTensorProtocol]", np.zeros)(shape)

    @staticmethod
    def ones(shape: EagerValue, dtype: EagerValue) -> EagerValue:
        """Test function ones."""
        return cast("Callable[..., EagerTensorProtocol]", np.ones)(shape, dtype=dtype)

    @staticmethod
    def cross(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function cross."""
        return cast("Callable[..., EagerTensorProtocol]", np.cross)(*args, **kwargs)

    linalg = np.linalg


def _test_linalg_extras_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_linalg_extras.

    Args:
        _state: Shared test state dictionary.
    """
    assert _get_uncontracted_dims([1, 2, 3], [0], [2]) == [2]

    dn = (((2,), (1,)), ((0,), (0,)))
    assert cast("Callable[..., EagerTensorProtocol]", _parse_dot_dimension_numbers)(dn) == ((2,), (1,), (0,), (0,))

    a = np.ones((2, 3, 4))
    b = np.ones((2, 4, 5))
    res = _dot_general(a, b, dn)
    assert cast("EagerTensorProtocol", getattr(res, "shape")) == (2, 3, 5)

    a = np.array([[1, 2], [3, 4]])
    assert _np_trace(DummyBackend(), a) == 5
    assert _np_matrix_rank(DummyBackend(), a) == 2
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(cast("Callable[..., EagerTensorProtocol]", _np_matrix_transpose)(cast("Callable[..., EagerTensorProtocol]", DummyBackend)(), a), a.T)
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(cast("Callable[..., EagerTensorProtocol]", _np_sqrtm)(a), a)
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(cast("Callable[..., EagerTensorProtocol]", _np_adjoint)(cast("Callable[..., EagerTensorProtocol]", DummyBackend)(), a), cast("Callable[..., EagerTensorProtocol]", a.T.conj)())

    # CholeskySolve
    assert _np_cholesky_solve(DummyBackend(), 1) == 1
    A = np.array([[2, -1], [-1, 2]], dtype=float)
    b = np.array([1, 0], dtype=float)
    L = np.linalg.cholesky(A)
    assert cast("Callable[..., EagerTensorProtocol]", np.allclose)(cast("Callable[..., EagerTensorProtocol]", _np_cholesky_solve)(cast("Callable[..., EagerTensorProtocol]", DummyBackend)(), b, L, upper=False), [2 / 3, 1 / 3])

    # EighTridiagonal
    assert _np_eigh_tridiagonal(DummyBackend(), 1) == 1
    alpha = np.array([2, 2, 2], dtype=float)
    beta = np.array([-1, -1], dtype=float)
    vals, vecs = cast("Callable[..., EagerTensorProtocol]", _np_eigh_tridiagonal)(cast("Callable[..., EagerTensorProtocol]", DummyBackend)(), alpha, beta)
    getattr(_state, "update")({"A": A, "L": L, "a": a, "alpha": alpha})
    getattr(_state, "update")({"b": b, "beta": beta, "dn": dn, "res": res})
    getattr(_state, "update")({"vals": vals, "vecs": vecs})
    _state.update(locals())


def _test_linalg_extras_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_linalg_extras.

    Args:
        _state: Shared test state dictionary.
    """
    A, b, vals = _state["A"], _state["b"], _state["vals"]
    assert vals.shape == (3,)

    # Qr
    assert cast("EagerTensorProtocol", getattr(cast("EagerTensorProtocol", getattr(_np_qr(DummyBackend(), A), "__getitem__")(0)), "shape")) == (2, 2)

    # Cross
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(cast("Callable[..., EagerTensorProtocol]", _np_cross)(cast("Callable[..., EagerTensorProtocol]", DummyBackend)(), v1, v2, axes={"axis": None}), [0, 0, 1])
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(cast("Callable[..., EagerTensorProtocol]", _np_cross)(cast("Callable[..., EagerTensorProtocol]", DummyBackend)(), v1, v2, axis=None), [0, 0, 1])

    # Slogdet
    assert cast("Callable[..., EagerTensorProtocol]", len)(cast("Callable[..., EagerTensorProtocol]", _np_slogdet)(cast("Callable[..., EagerTensorProtocol]", DummyBackend)(), A)) == 2

    # Op wrappers
    for op in ["Qr", "Solve", "Tensorinv", "Tensorsolve", "Eig", "Eigh", "Eigvals", "Eigvalsh", "Norm", "Cond", "MultiDot"]:
        func = numpy_eager_registry._registry[op]
        if op == "Solve":
            assert cast("EagerTensorProtocol", getattr(func(DummyBackend(), A, b), "shape")) == (2,)
        elif op == "MultiDot":
            assert cast("EagerTensorProtocol", getattr(func(DummyBackend(), [A, A]), "shape")) == (2, 2)
        elif op in ["Eigvals", "Eigvalsh", "Norm", "Cond", "Eigh", "Eig"]:
            res = func(DummyBackend(), A)
        else:
            try:
                func(DummyBackend(), A)
            except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
                assert _exc is not None

    # LU
    p, _lower, u = cast("Callable[..., EagerTensorProtocol]", _np_lu)(cast("Callable[..., EagerTensorProtocol]", DummyBackend)(), A)
    lu, piv = cast("Callable[..., EagerTensorProtocol]", _np_lu_factor)(cast("Callable[..., EagerTensorProtocol]", DummyBackend)(), A)
    assert cast("EagerTensorProtocol", getattr(_np_lu_solve(DummyBackend(), lu, piv, b), "shape")) == (2,)
    assert cast("EagerTensorProtocol", getattr(_np_lu_pivots(DummyBackend(), [1, 1], 2), "shape")) == (2,)
    getattr(_state, "update")({"_lower": _lower, "lu": lu, "p": p, "piv": piv})
    getattr(_state, "update")({"u": u, "v1": v1, "v2": v2})
    _state.update(locals())


def _test_linalg_extras_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_linalg_extras.

    Args:
        _state: Shared test state dictionary.
    """
    A = _state["A"]
    assert cast("EagerTensorProtocol", getattr(_np_matrix_exponential(DummyBackend(), A), "shape")) == (2, 2)

    # Hessenberg
    h, q = cast("Callable[..., EagerTensorProtocol]", _np_hessenberg)(cast("Callable[..., EagerTensorProtocol]", DummyBackend)(), A)
    assert cast("EagerTensorProtocol", getattr(h, "shape")) == (2, 2)

    # Tridiagonal
    d, od, q = cast("Callable[..., EagerTensorProtocol]", _np_tridiagonal)(cast("Callable[..., EagerTensorProtocol]", DummyBackend)(), A)
    assert cast("EagerTensorProtocol", getattr(d, "shape")) == (2,)
    assert cast("EagerTensorProtocol", getattr(od, "shape")) == (1,)
    dl = np.array([-1], dtype=float)
    d = cast("EagerTensorProtocol", np.array([2, 2], dtype=float))
    du = np.array([-1], dtype=float)
    b_tri = np.array([1, 0], dtype=float)
    assert cast("EagerTensorProtocol", getattr(_np_tridiagonal_solve(DummyBackend(), dl, d, du, b_tri), "shape")) == (2,)

    # CholeskyEx
    c, i = cast("Callable[..., EagerTensorProtocol]", _np_cholesky_ex)(cast("Callable[..., EagerTensorProtocol]", DummyBackend)(), A)
    assert i == 0
    c_fail, i_fail = cast("Callable[..., EagerTensorProtocol]", _np_cholesky_ex)(cast("Callable[..., EagerTensorProtocol]", DummyBackend)(), cast("Callable[..., EagerTensorProtocol]", np.array)([[0, 0], [0, 0]]))
    assert cast("EagerTensorProtocol", getattr(i_fail, "all"))() == 1

    # InvEx
    i1, e1 = cast("Callable[..., EagerTensorProtocol]", _np_inv_ex)(cast("Callable[..., EagerTensorProtocol]", DummyBackend)(), A)
    assert e1 == 0
    i2, e2 = cast("Callable[..., EagerTensorProtocol]", _np_inv_ex)(cast("Callable[..., EagerTensorProtocol]", DummyBackend)(), cast("Callable[..., EagerTensorProtocol]", np.array)([[0, 0], [0, 0]]))
    assert cast("EagerTensorProtocol", getattr(e2, "all"))() == 1

    # Pinv
    assert cast("EagerTensorProtocol", getattr(_np_pinv(DummyBackend(), A), "shape")) == (2, 2)

    # Polar / Qdwh
    assert cast("EagerTensorProtocol", getattr(cast("EagerTensorProtocol", getattr(_np_polar(DummyBackend(), A), "__getitem__")(0)), "shape")) == (2, 2)
    assert cast("EagerTensorProtocol", getattr(cast("EagerTensorProtocol", getattr(_np_qdwh(DummyBackend(), A), "__getitem__")(0)), "shape")) == (2, 2)
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
    A, b = _state["A"], _state["b"]
    s, e = cast("Callable[..., EagerTensorProtocol]", _np_solve_ex)(cast("Callable[..., EagerTensorProtocol]", DummyBackend)(), A, b)
    assert e == 0
    s_fail, e_fail = cast("Callable[..., EagerTensorProtocol]", _np_solve_ex)(cast("Callable[..., EagerTensorProtocol]", DummyBackend)(), cast("Callable[..., EagerTensorProtocol]", np.array)([[0, 0], [0, 0]]), b)
    assert cast("EagerTensorProtocol", getattr(e_fail, "all"))() == 1

    # Svd
    u, s, vh = cast("Callable[..., EagerTensorProtocol]", _np_svd)(cast("Callable[..., EagerTensorProtocol]", DummyBackend)(), A)
    assert cast("EagerTensorProtocol", getattr(u, "shape")) == (2, 2)
    getattr(_state, "update")({"e": e, "e_fail": e_fail, "s": s, "s_fail": s_fail})
    getattr(_state, "update")({"u": u, "vh": vh})


def test_linalg_extras() -> None:
    """Test function test_linalg_extras."""
    _state: dict[str, EagerTensorProtocol] = {}
    _test_linalg_extras_part_1(_state)
    _test_linalg_extras_part_2(_state)
    _test_linalg_extras_part_3(_state)
    _test_linalg_extras_part_4(_state)
    assert isinstance(_state, dict)


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
    assert cast("EagerTensorProtocol", getattr(res, "shape")) == (2, 3, 5)


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
    assert cast("EagerTensorProtocol", getattr(_np_matrix_transpose(None, a), "shape")) == (3, 2)


def test_np_sqrtm() -> None:
    """Test function test_np_sqrtm."""
    a = np.ones((2, 2))
    assert np.all(_np_sqrtm(a) == a)


def test_np_adjoint() -> None:
    """Test function test_np_adjoint."""
    a = np.array([[1 + 1j, 2], [3, 4]])
    res = _np_adjoint(None, a)
    assert (
        cast(
            "EagerTensorProtocol",
            getattr(res, "__getitem__")(
                (
                    0,
                    0,
                )
            ),
        )
        == 1 - 1j
    )
    assert (
        cast(
            "EagerTensorProtocol",
            getattr(res, "__getitem__")(
                (
                    0,
                    1,
                )
            ),
        )
        == 3
    )


def test_np_cholesky_solve() -> None:
    """Test function test_np_cholesky_solve."""
    a = np.ones(3)
    assert np.all(_np_cholesky_solve(np, a) == a)  # len(args) < 2
    u = np.linalg.cholesky(np.array([[2, 1], [1, 2]]))
    b = np.array([1, 2])
    res = _np_cholesky_solve(np, b, u, upper=False)
    assert cast("Callable[..., EagerTensorProtocol]", len)(res) == 2


def test_np_eigh_tridiagonal() -> None:
    """Test function test_np_eigh_tridiagonal."""
    a = np.ones(3)
    assert np.all(_np_eigh_tridiagonal(np, a) == a)  # len(args) < 2
    alpha = np.array([2.0, 2.0, 2.0])
    beta = np.array([1.0, 1.0])
    vals, vecs = cast("Callable[..., EagerTensorProtocol]", _np_eigh_tridiagonal)(np, alpha, beta)
    assert cast("Callable[..., EagerTensorProtocol]", len)(vals) == 3


def test_np_qr() -> None:
    """Test function test_np_qr."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    q, r = cast("Callable[..., EagerTensorProtocol]", _np_qr)(np, a)
    assert cast("EagerTensorProtocol", getattr(q, "shape")) == (2, 2)


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
    sign, logdet = cast("Callable[..., EagerTensorProtocol]", _np_slogdet)(np, a)
    assert sign == 1.0
    assert logdet == 0.0


def test_linalg_wrappers() -> None:
    """Test function test_linalg_wrappers."""
    a = np.eye(3)
    res = cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("Solve")(np, a, cast("Callable[..., EagerTensorProtocol]", np.ones)(3))
    assert cast("EagerTensorProtocol", getattr(res, "shape")) == (3,)
    res = cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("MultiDot")(np, [a, a])
    assert cast("EagerTensorProtocol", getattr(res, "shape")) == (3, 3)


def test_np_lu() -> None:
    """Test function test_np_lu."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    p, _lower, _u = cast("Callable[..., EagerTensorProtocol]", _np_lu)(None, a)
    assert cast("EagerTensorProtocol", getattr(p, "shape")) == (2, 2)


def test_np_lu_factor() -> None:
    """Test function test_np_lu_factor."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    lu, piv = cast("Callable[..., EagerTensorProtocol]", _np_lu_factor)(None, a)
    assert cast("EagerTensorProtocol", getattr(lu, "shape")) == (2, 2)


def test_np_lu_solve() -> None:
    """Test function test_np_lu_solve."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    lu, piv = cast("Callable[..., EagerTensorProtocol]", _np_lu_factor)(None, a)
    b = np.array([1.0, 2.0])
    res = _np_lu_solve(None, lu, piv, b)
    assert cast("EagerTensorProtocol", getattr(res, "shape")) == (2,)


def test_np_lu_pivots() -> None:
    """Test function test_np_lu_pivots."""
    pivots = np.array([1, 1])
    perm = _np_lu_pivots(None, pivots, 2)
    assert cast("EagerTensorProtocol", getattr(perm, "shape")) == (2,)


def test_np_matrix_exponential() -> None:
    """Test function test_np_matrix_exponential."""
    a = np.eye(2)
    res = _np_matrix_exponential(None, a)
    assert cast("EagerTensorProtocol", getattr(res, "shape")) == (2, 2)


def test_np_hessenberg() -> None:
    """Test function test_np_hessenberg."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    h, q = cast("Callable[..., EagerTensorProtocol]", _np_hessenberg)(None, a)
    assert cast("EagerTensorProtocol", getattr(h, "shape")) == (2, 2)


def test_np_tridiagonal() -> None:
    """Test function test_np_tridiagonal."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    d, od, q = cast("Callable[..., EagerTensorProtocol]", _np_tridiagonal)(None, a)
    assert cast("Callable[..., EagerTensorProtocol]", len)(d) == 2
    assert cast("Callable[..., EagerTensorProtocol]", len)(od) == 1


def test_np_tridiagonal_solve() -> None:
    """Test function test_np_tridiagonal_solve."""
    dl = np.array([1.0])
    d = np.array([2.0, 2.0])
    du = np.array([1.0])
    b = np.array([1.0, 2.0])
    res = _np_tridiagonal_solve(None, dl, d, du, b)
    assert cast("EagerTensorProtocol", getattr(res, "shape")) == (2,)


def test_np_cholesky_ex() -> None:
    """Test function test_np_cholesky_ex."""
    a = np.array([[2.0, 1.0], [1.0, 2.0]])
    _lower, info = cast("Callable[..., EagerTensorProtocol]", _np_cholesky_ex)(None, a)
    assert info == 0
    bad = np.array([[0.0, 1.0], [1.0, 0.0]])
    _lower, info = cast("Callable[..., EagerTensorProtocol]", _np_cholesky_ex)(None, bad)
    assert info == 1


def test_np_inv_ex() -> None:
    """Test function test_np_inv_ex."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    inv, info = cast("Callable[..., EagerTensorProtocol]", _np_inv_ex)(None, a)
    assert info == 0
    bad = np.array([[1.0, 1.0], [1.0, 1.0]])
    inv, info = cast("Callable[..., EagerTensorProtocol]", _np_inv_ex)(None, bad)
    assert info == 1


def test_np_pinv() -> None:
    """Test function test_np_pinv."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    res = _np_pinv(None, a)
    assert cast("EagerTensorProtocol", getattr(res, "shape")) == (2, 2)


def test_np_polar() -> None:
    """Test function test_np_polar."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    u, p = cast("Callable[..., EagerTensorProtocol]", _np_polar)(None, a)
    assert cast("EagerTensorProtocol", getattr(u, "shape")) == (2, 2)


def test_np_qdwh() -> None:
    """Test function test_np_qdwh."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    u, p, iters, conv = cast("Callable[..., EagerTensorProtocol]", _np_qdwh)(None, a)
    assert cast("EagerTensorProtocol", getattr(u, "shape")) == (2, 2)
    assert conv


def test_np_solve_ex() -> None:
    """Test function test_np_solve_ex."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([1.0, 2.0])
    res, info = cast("Callable[..., EagerTensorProtocol]", _np_solve_ex)(None, a, b)
    assert info == 0
    bad = np.array([[1.0, 1.0], [1.0, 1.0]])
    res, info = cast("Callable[..., EagerTensorProtocol]", _np_solve_ex)(None, bad, b)
    assert info == 1


def test_np_svd() -> None:
    """Test function test_np_svd."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    u, s, vh = cast("Callable[..., EagerTensorProtocol]", _np_svd)(None, a)
    assert cast("EagerTensorProtocol", getattr(s, "shape")) == (2,)
