"""Tests for test_core_math_arithmetic."""

from __future__ import annotations

import math
import typing

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
from tests.testing_utils import as_callable, as_mod, as_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


class _EmptyBackend:
    """Mock backend without any attributes."""

    pass


def _test_math_arithmetic_all_branches_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_math_arithmetic_all_branches.

    Args:
        _state: Shared test state dictionary.
    """
    empty_backend = _EmptyBackend()

    # 1. _true_divide
    # Case A: backend has divide
    class BackendDivide:
        """Backend with divide method."""

        def divide(self, a: float, b: float) -> float:
            """Divide a by b.

            Args:
                a: Test parameter `a`.
                b: Test parameter `b`.
            """
            return a / b

    assert _true_divide(as_mod(BackendDivide()), 6.0, 2.0) == 3.0

    # Case B: backend has true_divide (no divide)
    class BackendTrueDivide:
        """Backend with true_divide method."""

        def true_divide(self, a: float, b: float) -> float:
            """True divide a by b.

            Args:
                a: Test parameter `a`.
                b: Test parameter `b`.
            """
            return a / b

    assert _true_divide(as_mod(BackendTrueDivide()), 8.0, 2.0) == 4.0

    # Case C: backend has neither
    assert _true_divide(as_mod(empty_backend), 8.0, 2.0) is None

    # 2. _fmod
    # Case A: backend has fmod
    class BackendFmod:
        """Backend with fmod method."""

        def fmod(self, a: float, b: float) -> float:
            """Compute fmod.

            Args:
                a: Test parameter `a`.
                b: Test parameter `b`.
            """
            return math.fmod(a, b)

    assert _fmod(as_mod(BackendFmod()), 5.0, 3.0) == 2.0
    getattr(_state, "update")({"BackendDivide": BackendDivide, "BackendFmod": BackendFmod, "BackendTrueDivide": BackendTrueDivide, "empty_backend": empty_backend})
    _state.update(locals())


def _test_math_arithmetic_all_branches_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_math_arithmetic_all_branches.

    Args:
        _state: Shared test state dictionary.
    """
    empty_backend = _state["empty_backend"]

    class BackendRemainder:
        """Backend with remainder method."""

        def remainder(self, a: float, b: float) -> float:
            """Compute remainder.

            Args:
                a: Test parameter `a`.
                b: Test parameter `b`.
            """
            return a % b

    assert _fmod(as_mod(BackendRemainder()), 5.0, 3.0) == 2.0

    # Case C: backend has mod (no fmod, no remainder)
    class BackendMod:
        """Backend with mod method."""

        def mod(self, a: float, b: float) -> float:
            """Compute mod.

            Args:
                a: Test parameter `a`.
                b: Test parameter `b`.
            """
            return a % b

    assert _fmod(as_mod(BackendMod()), 5.0, 3.0) == 2.0

    # Case D: backend has none
    assert _fmod(empty_backend, 5.0, 3.0) is None

    # 3. _accumulate_n
    # Case A: args provided with non-empty list
    res_acc = _accumulate_n(empty_backend, [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])])
    assert as_callable(list)(res_acc) == [9, 12]

    # Case B: kwargs provided with inputs
    res_acc_kw = _accumulate_n(empty_backend, inputs=[np.array([10]), np.array([20])])
    assert as_callable(list)(res_acc_kw) == [30]

    # Case C: empty list in args
    empty_acc = _accumulate_n(empty_backend, [])
    assert as_tensor(getattr(empty_acc, "shape")) == ()

    # Case D: empty list in kwargs / no args
    empty_acc_kw = _accumulate_n(empty_backend)
    assert as_tensor(getattr(empty_acc_kw, "shape")) == ()

    # 4. _assign_add and _assign_sub
    assert _assign_add(empty_backend, 10, 5) == 15
    getattr(_state, "update")({"BackendMod": BackendMod, "BackendRemainder": BackendRemainder, "empty_acc": empty_acc, "empty_acc_kw": empty_acc_kw})
    getattr(_state, "update")({"res_acc": res_acc, "res_acc_kw": res_acc_kw})
    _state.update(locals())


def _test_math_arithmetic_all_branches_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_math_arithmetic_all_branches.

    Args:
        _state: Shared test state dictionary.
    """
    empty_backend = _state["empty_backend"]
    assert _assign_sub(empty_backend, 10, 5) == 5

    # 5. _add_n
    # Case A: args provided
    res_add_n = _add_n(empty_backend, [np.array([1, 1]), np.array([2, 2]), np.array([3, 3])])
    assert as_callable(list)(res_add_n) == [6, 6]

    # Case B: kwargs provided
    res_add_n_kw = _add_n(empty_backend, inputs=[np.array([5]), np.array([10])])
    assert as_callable(list)(res_add_n_kw) == [15]

    # Case C: empty list in args
    empty_add_n = _add_n(empty_backend, [])
    assert as_tensor(getattr(empty_add_n, "shape")) == ()

    # Case D: empty list in kwargs / no args
    empty_add_n_kw = _add_n(empty_backend)
    assert as_tensor(getattr(empty_add_n_kw, "shape")) == ()

    # 6. _modf
    class BackendModf:
        """Backend with modf method."""

        def modf(self, val: float) -> tuple[float, float]:
            """Compute modf.

            Args:
                val: Test parameter `val`.
            """
            return math.modf(val)

    fract, integral = as_callable(_modf)(as_callable(BackendModf)(), 2.5)
    assert as_callable(math.isclose)(fract, 0.5)
    assert as_callable(math.isclose)(integral, 2.0)

    # 7. _ravelmultiindex
    class BackendRavel:
        """Backend with ravel_multi_index."""

        def ravel_multi_index(self, multi_index: EagerValue, dims: EagerValue) -> EagerValue:
            """Ravel multi index.

            Args:
                multi_index: Test parameter `multi_index`.
                dims: Test parameter `dims`.
            """
            return as_callable(np.ravel_multi_index)(multi_index, dims)

    getattr(_state, "update")({"BackendModf": BackendModf, "BackendRavel": BackendRavel, "empty_add_n": empty_add_n, "empty_add_n_kw": empty_add_n_kw})
    getattr(_state, "update")({"fract": fract, "integral": integral, "res_add_n": res_add_n, "res_add_n_kw": res_add_n_kw})
    _state.update(locals())


def _test_math_arithmetic_all_branches_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_math_arithmetic_all_branches.

    Args:
        _state: Shared test state dictionary.
    """
    BackendRavel, empty_backend = _state["BackendRavel"], _state["empty_backend"]
    assert _ravelmultiindex(BackendRavel(), (np.array([1]), np.array([2])), (3, 4)) == 6

    # 8. _np_scattermul
    class BackendScattermul:
        """Backend with scattermul."""

        def scattermul(self, *args: EagerValue, **kwargs: EagerValue) -> str:
            """Scattermul.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "scattermul_result"

    assert _np_scattermul(as_mod(BackendScattermul()), "val") == "scattermul_result"
    assert _np_scattermul(empty_backend, "fallback_val") == "fallback_val"

    class BackendStringSubstr:
        """Backend with stringsubstr."""

        def stringsubstr(self, *args: EagerValue, **kwargs: EagerValue) -> str:
            """String substr.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "substr_result"

    assert _np_stringsubstr(as_mod(BackendStringSubstr()), "test") == "substr_result"
    substr_fb = _np_stringsubstr(empty_backend, ["hello", "world"])
    assert as_callable(list)(substr_fb) == ["hello", "world"]

    # 10. _np_tensorscattersub
    class BackendTensorScatterSub:
        """Backend with tensorscattersub."""

        def tensorscattersub(self, *args: EagerValue, **kwargs: EagerValue) -> str:
            """Tensor scatter sub.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "scattersub_result"

    assert _np_tensorscattersub(as_mod(BackendTensorScatterSub()), "target") == "scattersub_result"
    getattr(_state, "update")({"BackendScattermul": BackendScattermul, "BackendStringSubstr": BackendStringSubstr, "BackendTensorScatterSub": BackendTensorScatterSub, "substr_fb": substr_fb})
    _state.update(locals())


def _test_math_arithmetic_all_branches_part_5(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 5 of test_math_arithmetic_all_branches.

    Args:
        _state: Shared test state dictionary.
    """
    empty_backend = _state["empty_backend"]
    assert _np_tensorscattersub(empty_backend, "fallback_target") == "fallback_target"

    # 11. _np_truncatediv
    class BackendTruncatediv:
        """Backend with truncatediv."""

        def truncatediv(self, *args: EagerValue, **kwargs: EagerValue) -> str:
            """Truncate div.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "truncatediv_result"

    assert _np_truncatediv(as_mod(BackendTruncatediv()), 7.0, 2.0) == "truncatediv_result"
    trunc_fb = _np_truncatediv(empty_backend, np.array([7.5, -7.5]), np.array([2.0, 2.0]))
    assert as_callable(list)(trunc_fb) == [3.0, -3.0]

    class BackendTruncatemod:
        """Backend with truncatemod."""

        def truncatemod(self, *args: EagerValue, **kwargs: EagerValue) -> str:
            """Truncate mod.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "truncatemod_result"

    assert _np_truncatemod(as_mod(BackendTruncatemod()), 7.0, 2.0) == "truncatemod_result"
    mod_fb = _np_truncatemod(empty_backend, np.array([7.0, 8.0]), np.array([3.0, 3.0]))
    assert as_callable(list)(mod_fb) == [1.0, 2.0]

    # 13. _np_xdivy
    class BackendXdivy:
        """Backend with xdivy."""

        def xdivy(self, *args: EagerValue, **kwargs: EagerValue) -> str:
            """Xdivy.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "xdivy_result"

    getattr(_state, "update")({"BackendTruncatediv": BackendTruncatediv, "BackendTruncatemod": BackendTruncatemod, "BackendXdivy": BackendXdivy, "mod_fb": mod_fb})
    getattr(_state, "update")({"trunc_fb": trunc_fb})
    _state.update(locals())


def _test_math_arithmetic_all_branches_part_6(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 6 of test_math_arithmetic_all_branches.

    Args:
        _state: Shared test state dictionary.
    """
    BackendXdivy, empty_backend = _state["BackendXdivy"], _state["empty_backend"]
    assert _np_xdivy(BackendXdivy(), 0.0, 5.0) == "xdivy_result"
    xdivy_fb = _np_xdivy(empty_backend, np.array([0.0, 10.0]), np.array([5.0, 2.0]))
    assert as_callable(list)(xdivy_fb) == [0.0, 5.0]

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
    getattr(_state, "update")({"xdivy_fb": xdivy_fb})


def test_math_arithmetic_all_branches() -> None:
    """Test 100% lines and branches of math_arithmetic.py.

    Returns:
        None
    """
    _state: dict[str, EagerTensorProtocol] = {}
    _test_math_arithmetic_all_branches_part_1(_state)
    _test_math_arithmetic_all_branches_part_2(_state)
    _test_math_arithmetic_all_branches_part_3(_state)
    _test_math_arithmetic_all_branches_part_4(_state)
    _test_math_arithmetic_all_branches_part_5(_state)
    _test_math_arithmetic_all_branches_part_6(_state)
    assert len(_state) > 0
