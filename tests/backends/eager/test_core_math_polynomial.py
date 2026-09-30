"""Tests for test_core_math_polynomial."""

from __future__ import annotations

from typing import Any

from ml_switcheroo_compiler.backends.eager.core_math_ops.math_polynomial import (
    _np_polyint,
    _polyval,
)
from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry


def test_math_polynomial_coverage() -> None:
    """Test 100% lines and branches of math_polynomial.

    Returns:
        None
    """

    # _polyval
    # Case A: backend has polyval
    class PolyvalBackend:
        """Backend with polyval."""

        def polyval(self, *args: Any, **kwargs: Any) -> str:
            """Mock polyval."""
            return "polyval_val"

    assert _polyval(PolyvalBackend(), [1, 2], 3) == "polyval_val"

    # Case B: backend lacks polyval (returns None)
    class EmptyBackend:
        """Empty backend."""

        pass

    assert _polyval(EmptyBackend(), [1, 2], 3) is None

    # _np_polyint
    # Case A: backend has polyint
    class PolyintBackend:
        """Backend with polyint."""

        def polyint(self, *args: Any, **kwargs: Any) -> str:
            """Mock polyint."""
            return "polyint_val"

    assert _np_polyint(PolyintBackend(), [1, 2]) == "polyint_val"

    # Case B: backend lacks polyint -> fallback to np.polyint
    p_int = _np_polyint(EmptyBackend(), [3, 2, 1])
    assert list(p_int) == [1.0, 1.0, 1.0, 0.0]

    # Registry
    assert global_eager_registry.get("Polyval") is not None
    assert global_eager_registry.get("Polyint") is not None
