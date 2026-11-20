"""Tests for test_core_math_polynomial."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.eager.core_math_ops.math_polynomial import _np_polyint, _polyval
from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry
from tests.testing_utils import MockTestStub, coerce_operand_list

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def test_math_polynomial_coverage() -> None:
    """Test 100% lines and branches of math_polynomial.

    Returns:
        None
    """

    # _polyval
    # Case A: backend has polyval
    class PolyvalBackend(MockTestStub):
        """Backend with polyval."""

        def polyval(self, *args: BackendArg, **kwargs: BackendArg) -> str:
            """Mock polyval.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                str: Computed output value from `polyval`.
            """
            return "polyval_val"

    assert _polyval(PolyvalBackend(), [1, 2], 3) == "polyval_val"

    # Case B: backend lacks polyval (returns None)
    class EmptyBackend(MockTestStub):
        """Empty backend."""

        pass

    assert _polyval(EmptyBackend(), [1, 2], 3) is None

    # _np_polyint
    # Case A: backend has polyint
    class PolyintBackend(MockTestStub):
        """Backend with polyint."""

        def polyint(self, *args: BackendArg, **kwargs: BackendArg) -> str:
            """Mock polyint.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                str: Computed output value from `polyint`.
            """
            return "polyint_val"

    assert _np_polyint(PolyintBackend(), [1, 2]) == "polyint_val"

    # Case B: backend lacks polyint -> fallback to np.polyint
    p_int = _np_polyint(EmptyBackend(), [3, 2, 1])
    assert coerce_operand_list(p_int) == [1.0, 1.0, 1.0, 0.0]

    # Registry
    assert global_eager_registry.get("Polyval") is not None
    assert global_eager_registry.get("Polyint") is not None
