"""Tests for test_core_math_polynomial."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.eager.core_math_ops.math_polynomial import (
    _np_polyint,
    _polyval,
)
from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry
from tests.testing_utils import as_callable, as_mod

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def test_math_polynomial_coverage() -> None:
    """Test 100% lines and branches of math_polynomial.

    Returns:
        None
    """

    # _polyval
    # Case A: backend has polyval
    class PolyvalBackend:
        """Backend with polyval."""

        def polyval(self, *args: EagerValue, **kwargs: EagerValue) -> str:
            """Mock polyval.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "polyval_val"

    assert _polyval(as_mod(PolyvalBackend()), [1, 2], 3) == "polyval_val"

    # Case B: backend lacks polyval (returns None)
    class EmptyBackend:
        """Empty backend."""

        pass

    assert _polyval(as_mod(EmptyBackend()), [1, 2], 3) is None

    # _np_polyint
    # Case A: backend has polyint
    class PolyintBackend:
        """Backend with polyint."""

        def polyint(self, *args: EagerValue, **kwargs: EagerValue) -> str:
            """Mock polyint.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "polyint_val"

    assert _np_polyint(as_mod(PolyintBackend()), [1, 2]) == "polyint_val"

    # Case B: backend lacks polyint -> fallback to np.polyint
    p_int = _np_polyint(as_mod(EmptyBackend()), [3, 2, 1])
    assert as_callable(list)(p_int) == [1.0, 1.0, 1.0, 0.0]

    # Registry
    assert global_eager_registry.get("Polyval") is not None
    assert global_eager_registry.get("Polyint") is not None
