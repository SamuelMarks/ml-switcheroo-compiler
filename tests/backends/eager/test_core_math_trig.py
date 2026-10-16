"""Tests for test_core_math_trig."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np

from ml_switcheroo_compiler.backends.eager.core_math_ops.math_trig import (
    _acos,
    _acosh,
    _asin,
    _asinh,
    _atan,
    _atan2,
    _atanh,
    _isin,
    _isinf,
    _isposinf,
    _sinc,
)
from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


class _EmptyBackend:
    """Mock backend without any attributes."""

    pass


def _test_math_trig_all_branches_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_math_trig_all_branches.

    Args:
        _state: Shared test state dictionary.
    """
    empty_backend = _EmptyBackend()

    # 1. _acos
    # Case A: backend has arccos
    class BackendArccos:
        """Backend with arccos."""

        def arccos(self, x: float) -> str:
            """Arccos."""
            return "arccos_val"

    assert _acos(BackendArccos(), 0.5) == "arccos_val"

    # Case B: backend has acos (no arccos)
    class BackendAcos:
        """Backend with acos."""

        def acos(self, x: float) -> str:
            """Acos."""
            return "acos_val"

    assert _acos(BackendAcos(), 0.5) == "acos_val"

    # Case C: backend has neither -> fallback to np.arccos
    assert cast("Callable[..., EagerTensorProtocol]", np.isclose)(cast("Callable[..., EagerTensorProtocol]", _acos)(empty_backend, 0.5), cast("Callable[..., EagerTensorProtocol]", math.acos)(0.5))

    # 2. _acosh
    # Case A: backend has arccosh
    class BackendArccosh:
        """Backend with arccosh."""

        def arccosh(self, x: float) -> str:
            """Arccosh."""
            return "arccosh_val"

    assert _acosh(BackendArccosh(), 1.5) == "arccosh_val"
    getattr(_state, "update")({"BackendAcos": BackendAcos, "BackendArccos": BackendArccos, "BackendArccosh": BackendArccosh, "empty_backend": empty_backend})
    _state.update(locals())


def _test_math_trig_all_branches_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_math_trig_all_branches.

    Args:
        _state: Shared test state dictionary.
    """
    empty_backend = _state["empty_backend"]

    class BackendAcosh:
        """Backend with acosh."""

        def acosh(self, x: float) -> str:
            """Acosh."""
            return "acosh_val"

    assert _acosh(BackendAcosh(), 1.5) == "acosh_val"

    # Case C: fallback
    assert cast("Callable[..., EagerTensorProtocol]", np.isclose)(cast("Callable[..., EagerTensorProtocol]", _acosh)(empty_backend, 0.5), cast("Callable[..., EagerTensorProtocol]", math.acos)(0.5))

    # 3. _asin
    # Case A: backend has arcsin
    class BackendArcsin:
        """Backend with arcsin."""

        def arcsin(self, x: float) -> str:
            """Arcsin."""
            return "arcsin_val"

    assert _asin(BackendArcsin(), 0.5) == "arcsin_val"

    # Case B: backend has asin
    class BackendAsin:
        """Backend with asin."""

        def asin(self, x: float) -> str:
            """Asin."""
            return "asin_val"

    assert _asin(BackendAsin(), 0.5) == "asin_val"

    # Case C: fallback
    assert cast("Callable[..., EagerTensorProtocol]", np.isclose)(cast("Callable[..., EagerTensorProtocol]", _asin)(empty_backend, 0.5), cast("Callable[..., EagerTensorProtocol]", math.asin)(0.5))
    getattr(_state, "update")({"BackendAcosh": BackendAcosh, "BackendArcsin": BackendArcsin, "BackendAsin": BackendAsin})
    _state.update(locals())


def _test_math_trig_all_branches_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_math_trig_all_branches.

    Args:
        _state: Shared test state dictionary.
    """
    empty_backend = _state["empty_backend"]

    class BackendArcsinh:
        """Backend with arcsinh."""

        def arcsinh(self, x: float) -> str:
            """Arcsinh."""
            return "arcsinh_val"

    assert _asinh(BackendArcsinh(), 0.5) == "arcsinh_val"

    class BackendAsin:
        """Backend with asinh."""

        def asinh(self, x: float) -> str:
            """Asinh."""
            return "asinh_val"

    assert _asinh(BackendAsin(), 0.5) == "asinh_val"

    # Case C: fallback
    assert cast("Callable[..., EagerTensorProtocol]", np.isclose)(cast("Callable[..., EagerTensorProtocol]", _asinh)(empty_backend, 0.5), cast("Callable[..., EagerTensorProtocol]", math.asin)(0.5))

    # 5. _atan
    # Case A: backend has arctan
    class BackendArctan:
        """Backend with arctan."""

        def arctan(self, x: float) -> str:
            """Arctan."""
            return "arctan_val"

    assert _atan(BackendArctan(), 0.5) == "arctan_val"
    getattr(_state, "update")({"BackendArcsinh": BackendArcsinh, "BackendArctan": BackendArctan, "BackendAsin": BackendAsin})
    _state.update(locals())


def _test_math_trig_all_branches_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_math_trig_all_branches.

    Args:
        _state: Shared test state dictionary.
    """
    empty_backend = _state["empty_backend"]

    class BackendAtan:
        """Backend with atan."""

        def atan(self, x: float) -> str:
            """Atan."""
            return "atan_val"

    assert _atan(BackendAtan(), 0.5) == "atan_val"

    # Case C: fallback
    assert cast("Callable[..., EagerTensorProtocol]", np.isclose)(cast("Callable[..., EagerTensorProtocol]", _atan)(empty_backend, 0.5), cast("Callable[..., EagerTensorProtocol]", math.atan)(0.5))

    # 6. _atanh
    # Case A: backend has arctanh
    class BackendArctanh:
        """Backend with arctanh."""

        def arctanh(self, x: float) -> str:
            """Arctanh."""
            return "arctanh_val"

    assert _atanh(BackendArctanh(), 0.5) == "arctanh_val"

    # Case B: backend has atanh
    class BackendAtanh:
        """Backend with atanh."""

        def atanh(self, x: float) -> str:
            """Atanh."""
            return "atanh_val"

    assert _atanh(BackendAtanh(), 0.5) == "atanh_val"

    # Case C: fallback
    assert cast("Callable[..., EagerTensorProtocol]", np.isclose)(cast("Callable[..., EagerTensorProtocol]", _atanh)(empty_backend, 0.5), cast("Callable[..., EagerTensorProtocol]", math.atan)(0.5))
    getattr(_state, "update")({"BackendArctanh": BackendArctanh, "BackendAtan": BackendAtan, "BackendAtanh": BackendAtanh})
    _state.update(locals())


def _test_math_trig_all_branches_part_5(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 5 of test_math_trig_all_branches.

    Args:
        _state: Shared test state dictionary.
    """
    empty_backend = _state["empty_backend"]

    class BackendArctan2:
        """Backend with arctan2."""

        def arctan2(self, y: float, x: float) -> str:
            """Arctan2."""
            return "arctan2_val"

    assert _atan2(BackendArctan2(), 1.0, 1.0) == "arctan2_val"

    # Case B: backend has atan2
    class BackendAtan2:
        """Backend with atan2."""

        def atan2(self, y: float, x: float) -> str:
            """Atan2."""
            return "atan2_val"

    assert _atan2(BackendAtan2(), 1.0, 1.0) == "atan2_val"

    # Case C: fallback
    assert cast("Callable[..., EagerTensorProtocol]", np.isclose)(cast("Callable[..., EagerTensorProtocol]", _atan2)(empty_backend, 1.0, 1.0), cast("Callable[..., EagerTensorProtocol]", math.atan2)(1.0, 1.0))

    class BackendSinc:
        """Backend with sinc."""

        def sinc(self, x: float) -> str:
            """Sinc."""
            return "sinc_val"

    assert _sinc(BackendSinc(), 0.5) == "sinc_val"

    # Case B: fallback
    assert cast("Callable[..., EagerTensorProtocol]", np.isclose)(cast("Callable[..., EagerTensorProtocol]", _sinc)(empty_backend, 0.0), 1.0)
    getattr(_state, "update")({"BackendArctan2": BackendArctan2, "BackendAtan2": BackendAtan2, "BackendSinc": BackendSinc})
    _state.update(locals())


def _test_math_trig_all_branches_part_6(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 6 of test_math_trig_all_branches.

    Args:
        _state: Shared test state dictionary.
    """
    empty_backend = _state["empty_backend"]

    class BackendIsin:
        """Backend with isin."""

        def isin(self, element: EagerValue, test_elements: EagerValue) -> str:
            """Isin."""
            return "isin_val"

    assert _isin(BackendIsin(), [1], [1, 2]) == "isin_val"

    # 10. _isinf
    # Case A: backend has isinf and args are passed directly
    class BackendIsinfDirect:
        """Backend with isinf."""

        def isinf(self, x: EagerValue) -> str:
            """Isinf."""
            return "isinf_direct"

    assert _isinf(BackendIsinfDirect(), np.array([float("inf")])) == "isinf_direct"

    # Case B: backend lacks isinf -> fallback to np.isinf
    inf_res = _isinf(empty_backend, np.array([float("inf"), 1.0]))
    assert cast("Callable[..., EagerTensorProtocol]", list)(inf_res) == [True, False]

    class BackendIsposinfDirect:
        """Backend with isposinf."""

        def isposinf(self, x: EagerValue) -> str:
            """Isposinf."""
            return "isposinf_direct"

    assert _isposinf(BackendIsposinfDirect(), np.array([float("inf")])) == "isposinf_direct"

    # Case B: backend lacks isposinf -> fallback to np.isposinf
    posinf_res = _isposinf(empty_backend, np.array([float("inf"), float("-inf"), 1.0]))
    assert cast("Callable[..., EagerTensorProtocol]", list)(posinf_res) == [True, False, False]
    getattr(_state, "update")({"BackendIsin": BackendIsin, "BackendIsinfDirect": BackendIsinfDirect, "BackendIsposinfDirect": BackendIsposinfDirect, "inf_res": inf_res})
    getattr(_state, "update")({"posinf_res": posinf_res})
    _state.update(locals())


def _test_math_trig_all_branches_part_7(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 7 of test_math_trig_all_branches.

    Args:
        _state: Shared test state dictionary.
    """
    for op_name in [
        "Acos",
        "Acosh",
        "Asin",
        "Asinh",
        "Atan",
        "Atanh",
        "Atan2",
        "Sinc",
        "Isin",
        "Isinf",
        "Isposinf",
    ]:
        assert global_eager_registry.get(op_name) is not None


def test_math_trig_all_branches() -> None:
    """Test 100% lines and branches of math_trig.py.

    Returns:
        None
    """
    _state: dict[str, EagerTensorProtocol] = {}
    _test_math_trig_all_branches_part_1(_state)
    _test_math_trig_all_branches_part_2(_state)
    _test_math_trig_all_branches_part_3(_state)
    _test_math_trig_all_branches_part_4(_state)
    _test_math_trig_all_branches_part_5(_state)
    _test_math_trig_all_branches_part_6(_state)
    _test_math_trig_all_branches_part_7(_state)
    assert isinstance(_state, dict)
