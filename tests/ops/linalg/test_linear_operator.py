"""Tests for test_linear_operator."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol


def test_linear_operator_shape_inference() -> None:
    """Test function test_linear_operator_shape_inference."""
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperator

    assert LinearOperator().infer_shape() == ()

    class MockOperand:
        """Test class MockOperand."""

        shape = (1, 2)

    assert cast("EagerCallableProtocol", LinearOperator)().infer_shape(cast("EagerCallableProtocol", MockOperand)()) == (1, 2)
    assert cast("EagerCallableProtocol", LinearOperator)().infer_shape(operand=cast("EagerCallableProtocol", MockOperand)()) == (1, 2)
