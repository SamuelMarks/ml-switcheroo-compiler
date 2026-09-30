"""Tests for test_core_math_arithmetic."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ml_switcheroo_compiler.backends.eager.core_math_ops.math_arithmetic import (
    _accumulate_n,
    _add_n,
    _assign_add,
    _assign_sub,
    _fmod,
    _modf,
    _np_scattermul,
    _np_stringsubstr,
    _np_tensorscattersub,
    _np_truncatediv,
    _np_truncatemod,
    _np_xdivy,
    _ravelmultiindex,
    _true_divide,
)
from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry


class _EmptyBackend:
    """Mock backend without any attributes."""

    pass


def test_math_arithmetic_all_branches() -> None:
    """Test 100% lines and branches of math_arithmetic.py.

    Returns:
        None
    """
    empty_backend = _EmptyBackend()

    # 1. _true_divide
    # Case A: backend has divide
    class BackendDivide:
        """Backend with divide method."""

        def divide(self, a: float, b: float) -> float:
            """Divide a by b."""
            return a / b

    assert _true_divide(BackendDivide(), 6.0, 2.0) == 3.0

    # Case B: backend has true_divide (no divide)
    class BackendTrueDivide:
        """Backend with true_divide method."""

        def true_divide(self, a: float, b: float) -> float:
            """True divide a by b."""
            return a / b

    assert _true_divide(BackendTrueDivide(), 8.0, 2.0) == 4.0

    # Case C: backend has neither
    assert _true_divide(empty_backend, 8.0, 2.0) is None

    # 2. _fmod
    # Case A: backend has fmod
    class BackendFmod:
        """Backend with fmod method."""

        def fmod(self, a: float, b: float) -> float:
            """Compute fmod."""
            return math.fmod(a, b)

    assert _fmod(BackendFmod(), 5.0, 3.0) == 2.0

    # Case B: backend has remainder (no fmod)
    class BackendRemainder:
        """Backend with remainder method."""

        def remainder(self, a: float, b: float) -> float:
            """Compute remainder."""
            return a % b

    assert _fmod(BackendRemainder(), 5.0, 3.0) == 2.0

    # Case C: backend has mod (no fmod, no remainder)
    class BackendMod:
        """Backend with mod method."""

        def mod(self, a: float, b: float) -> float:
            """Compute mod."""
            return a % b

    assert _fmod(BackendMod(), 5.0, 3.0) == 2.0

    # Case D: backend has none
    assert _fmod(empty_backend, 5.0, 3.0) is None

    # 3. _accumulate_n
    # Case A: args provided with non-empty list
    res_acc = _accumulate_n(empty_backend, [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])])
    assert list(res_acc) == [9, 12]

    # Case B: kwargs provided with inputs
    res_acc_kw = _accumulate_n(empty_backend, inputs=[np.array([10]), np.array([20])])
    assert list(res_acc_kw) == [30]

    # Case C: empty list in args
    empty_acc = _accumulate_n(empty_backend, [])
    assert empty_acc.shape == ()

    # Case D: empty list in kwargs / no args
    empty_acc_kw = _accumulate_n(empty_backend)
    assert empty_acc_kw.shape == ()

    # 4. _assign_add and _assign_sub
    assert _assign_add(empty_backend, 10, 5) == 15
    assert _assign_sub(empty_backend, 10, 5) == 5

    # 5. _add_n
    # Case A: args provided
    res_add_n = _add_n(empty_backend, [np.array([1, 1]), np.array([2, 2]), np.array([3, 3])])
    assert list(res_add_n) == [6, 6]

    # Case B: kwargs provided
    res_add_n_kw = _add_n(empty_backend, inputs=[np.array([5]), np.array([10])])
    assert list(res_add_n_kw) == [15]

    # Case C: empty list in args
    empty_add_n = _add_n(empty_backend, [])
    assert empty_add_n.shape == ()

    # Case D: empty list in kwargs / no args
    empty_add_n_kw = _add_n(empty_backend)
    assert empty_add_n_kw.shape == ()

    # 6. _modf
    class BackendModf:
        """Backend with modf method."""

        def modf(self, val: float) -> tuple[float, float]:
            """Compute modf."""
            return math.modf(val)

    fract, integral = _modf(BackendModf(), 2.5)
    assert math.isclose(fract, 0.5)
    assert math.isclose(integral, 2.0)

    # 7. _ravelmultiindex
    class BackendRavel:
        """Backend with ravel_multi_index."""

        def ravel_multi_index(self, multi_index: Any, dims: Any) -> Any:
            """Ravel multi index."""
            return np.ravel_multi_index(multi_index, dims)

    assert _ravelmultiindex(BackendRavel(), (np.array([1]), np.array([2])), (3, 4)) == 6

    # 8. _np_scattermul
    class BackendScattermul:
        """Backend with scattermul."""

        def scattermul(self, *args: Any, **kwargs: Any) -> str:
            """Scattermul."""
            return "scattermul_result"

    assert _np_scattermul(BackendScattermul(), "val") == "scattermul_result"
    assert _np_scattermul(empty_backend, "fallback_val") == "fallback_val"

    # 9. _np_stringsubstr
    class BackendStringSubstr:
        """Backend with stringsubstr."""

        def stringsubstr(self, *args: Any, **kwargs: Any) -> str:
            """String substr."""
            return "substr_result"

    assert _np_stringsubstr(BackendStringSubstr(), "test") == "substr_result"
    substr_fb = _np_stringsubstr(empty_backend, ["hello", "world"])
    assert list(substr_fb) == ["hello", "world"]

    # 10. _np_tensorscattersub
    class BackendTensorScatterSub:
        """Backend with tensorscattersub."""

        def tensorscattersub(self, *args: Any, **kwargs: Any) -> str:
            """Tensor scatter sub."""
            return "scattersub_result"

    assert _np_tensorscattersub(BackendTensorScatterSub(), "target") == "scattersub_result"
    assert _np_tensorscattersub(empty_backend, "fallback_target") == "fallback_target"

    # 11. _np_truncatediv
    class BackendTruncatediv:
        """Backend with truncatediv."""

        def truncatediv(self, *args: Any, **kwargs: Any) -> str:
            """Truncate div."""
            return "truncatediv_result"

    assert _np_truncatediv(BackendTruncatediv(), 7.0, 2.0) == "truncatediv_result"
    trunc_fb = _np_truncatediv(empty_backend, np.array([7.5, -7.5]), np.array([2.0, 2.0]))
    assert list(trunc_fb) == [3.0, -3.0]

    # 12. _np_truncatemod
    class BackendTruncatemod:
        """Backend with truncatemod."""

        def truncatemod(self, *args: Any, **kwargs: Any) -> str:
            """Truncate mod."""
            return "truncatemod_result"

    assert _np_truncatemod(BackendTruncatemod(), 7.0, 2.0) == "truncatemod_result"
    mod_fb = _np_truncatemod(empty_backend, np.array([7.0, 8.0]), np.array([3.0, 3.0]))
    assert list(mod_fb) == [1.0, 2.0]

    # 13. _np_xdivy
    class BackendXdivy:
        """Backend with xdivy."""

        def xdivy(self, *args: Any, **kwargs: Any) -> str:
            """Xdivy."""
            return "xdivy_result"

    assert _np_xdivy(BackendXdivy(), 0.0, 5.0) == "xdivy_result"
    xdivy_fb = _np_xdivy(empty_backend, np.array([0.0, 10.0]), np.array([5.0, 2.0]))
    assert list(xdivy_fb) == [0.0, 5.0]

    # Verify registration in global_eager_registry
    for op_name in [
        "TrueDivide",
        "Fmod",
        "AccumulateN",
        "AssignAdd",
        "AssignSub",
        "AddN",
        "Modf",
        "RavelMultiIndex",
        "ScatterMul",
        "StringSubstr",
        "TensorScatterSub",
        "TruncateDiv",
        "TruncateMod",
        "Xdivy",
    ]:
        assert global_eager_registry.get(op_name) is not None
