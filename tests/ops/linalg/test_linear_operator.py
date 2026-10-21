"""Tests for test_linear_operator."""

from __future__ import annotations

import typing

from tests.testing_utils import as_callable

if typing.TYPE_CHECKING:
    pass


def test_linear_operator_shape_inference() -> None:
    """Test function test_linear_operator_shape_inference."""
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperator

    assert LinearOperator().infer_shape() == ()

    class MockOperand:
        """Test class MockOperand."""

        shape = (1, 2)

    assert as_callable(LinearOperator)().infer_shape(as_callable(MockOperand)()) == (1, 2)
    assert as_callable(LinearOperator)().infer_shape(operand=as_callable(MockOperand)()) == (1, 2)
