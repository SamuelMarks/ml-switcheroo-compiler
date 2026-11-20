"""Behavioral unit tests for state."""

from __future__ import annotations

from ml_switcheroo_compiler.ops.state import AssignVariable, ReadVariable


def test_state_infer_shape() -> None:
    """Test function test_state_infer_shape."""
    op1 = ReadVariable()
    assert op1.infer_shape(shape=(2, 3)) == (2, 3)
    assert op1.infer_shape() == ()

    op2 = AssignVariable()
    assert op2.infer_shape((2, 3)) == (2, 3)


def test_state_ops_shape_inference() -> None:
    """Verify ReadVariable and AssignVariable shape inference."""
    read_op = ReadVariable()
    assert read_op.infer_shape(shape=(2, 3)) == (2, 3)
    assert read_op.infer_shape(shape="batch_size") == "batch_size"
    assert read_op.infer_shape(shape=123) == ()
    assert read_op.infer_shape() == ()

    assign_op = AssignVariable()
    assert assign_op.infer_shape((4, 5, 6)) == (4, 5, 6)
    assert assign_op.infer_shape("tensor_symbol") == "tensor_symbol"
