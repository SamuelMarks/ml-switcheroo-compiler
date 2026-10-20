"""Tests for test_aggregations."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.ops.reductions.aggregations import NaryMathOp

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerValue


class MockTensor:
    """Test class MockTensor."""

    def __init__(self, shape: EagerValue) -> None:
        """Test function __init__.

        Args:
            shape: Test parameter `shape`.
        """
        self.shape = shape


def test_nary_math_op_infer_shape() -> None:
    """Test function test_nary_math_op_infer_shape."""
    op = NaryMathOp()
    t1 = MockTensor((2, 3))
    t2 = MockTensor((2, 3))

    # Empty args
    assert op.infer_shape() == ()

    # Empty list
    assert op.infer_shape([]) == ()
    assert op.infer_shape(inputs=[]) == ()

    # Valid list
    assert typing.cast("EagerCallableProtocol", op.infer_shape)([t1, t2]) == (2, 3)
    assert typing.cast("EagerCallableProtocol", op.infer_shape)(inputs=[t1, t2]) == (2, 3)

    # Not a list
    assert typing.cast("EagerCallableProtocol", op.infer_shape)(t1) == ()
