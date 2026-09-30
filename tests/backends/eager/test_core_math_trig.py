"""Tests for test_core_math_trig."""

from __future__ import annotations

import math
from typing import Any

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


class _EmptyBackend:
    """Mock backend without any attributes."""

    pass


def test_math_trig_all_branches() -> None:
    """Test 100% lines and branches of math_trig.py.

    Returns:
        None
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
    assert np.isclose(_acos(empty_backend, 0.5), math.acos(0.5))

    # 2. _acosh
    # Case A: backend has arccosh
    class BackendArccosh:
        """Backend with arccosh."""

        def arccosh(self, x: float) -> str:
            """Arccosh."""
            return "arccosh_val"

    assert _acosh(BackendArccosh(), 1.5) == "arccosh_val"

    # Case B: backend has acosh (no arccosh)
    class BackendAcosh:
        """Backend with acosh."""

        def acosh(self, x: float) -> str:
            """Acosh."""
            return "acosh_val"

    assert _acosh(BackendAcosh(), 1.5) == "acosh_val"

    # Case C: fallback
    assert np.isclose(_acosh(empty_backend, 0.5), math.acos(0.5))

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
    assert np.isclose(_asin(empty_backend, 0.5), math.asin(0.5))

    # 4. _asinh
    # Case A: backend has arcsinh
    class BackendArcsinh:
        """Backend with arcsinh."""

        def arcsinh(self, x: float) -> str:
            """Arcsinh."""
            return "arcsinh_val"

    assert _asinh(BackendArcsinh(), 0.5) == "arcsinh_val"

    # Case B: backend has asinh
    class BackendAsin:
        """Backend with asinh."""

        def asinh(self, x: float) -> str:
            """Asinh."""
            return "asinh_val"

    assert _asinh(BackendAsin(), 0.5) == "asinh_val"

    # Case C: fallback
    assert np.isclose(_asinh(empty_backend, 0.5), math.asin(0.5))

    # 5. _atan
    # Case A: backend has arctan
    class BackendArctan:
        """Backend with arctan."""

        def arctan(self, x: float) -> str:
            """Arctan."""
            return "arctan_val"

    assert _atan(BackendArctan(), 0.5) == "arctan_val"

    # Case B: backend has atan
    class BackendAtan:
        """Backend with atan."""

        def atan(self, x: float) -> str:
            """Atan."""
            return "atan_val"

    assert _atan(BackendAtan(), 0.5) == "atan_val"

    # Case C: fallback
    assert np.isclose(_atan(empty_backend, 0.5), math.atan(0.5))

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
    assert np.isclose(_atanh(empty_backend, 0.5), math.atan(0.5))

    # 7. _atan2
    # Case A: backend has arctan2
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
    assert np.isclose(_atan2(empty_backend, 1.0, 1.0), math.atan2(1.0, 1.0))

    # 8. _sinc
    # Case A: backend has sinc
    class BackendSinc:
        """Backend with sinc."""

        def sinc(self, x: float) -> str:
            """Sinc."""
            return "sinc_val"

    assert _sinc(BackendSinc(), 0.5) == "sinc_val"

    # Case B: fallback
    assert np.isclose(_sinc(empty_backend, 0.0), 1.0)

    # 9. _isin
    class BackendIsin:
        """Backend with isin."""

        def isin(self, element: Any, test_elements: Any) -> str:
            """Isin."""
            return "isin_val"

    assert _isin(BackendIsin(), [1], [1, 2]) == "isin_val"

    # 10. _isinf
    # Case A: backend has isinf and args are passed directly
    class BackendIsinfDirect:
        """Backend with isinf."""

        def isinf(self, x: Any) -> str:
            """Isinf."""
            return "isinf_direct"

    assert _isinf(BackendIsinfDirect(), np.array([float("inf")])) == "isinf_direct"

    # Case B: backend lacks isinf -> fallback to np.isinf
    inf_res = _isinf(empty_backend, np.array([float("inf"), 1.0]))
    assert list(inf_res) == [True, False]

    # 11. _isposinf
    # Case A: backend has isposinf directly
    class BackendIsposinfDirect:
        """Backend with isposinf."""

        def isposinf(self, x: Any) -> str:
            """Isposinf."""
            return "isposinf_direct"

    assert _isposinf(BackendIsposinfDirect(), np.array([float("inf")])) == "isposinf_direct"

    # Case B: backend lacks isposinf -> fallback to np.isposinf
    posinf_res = _isposinf(empty_backend, np.array([float("inf"), float("-inf"), 1.0]))
    assert list(posinf_res) == [True, False, False]

    # Registry verification
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
