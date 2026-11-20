"""Behavioral unit tests for raw ops."""

from __future__ import annotations

from ml_switcheroo_compiler.ops.raw_ops import RawConv2D, RawMatMul, RawMerge, RawOp, RawSwitch


def test_raw_op_infer_shape() -> None:
    """Test function test_raw_op_infer_shape."""
    op = RawOp()
    assert op.infer_shape() == ()
    assert op.infer_shape(1, 2, a=3) == ()


def test_raw_op_registry() -> None:
    """Test function test_raw_op_registry."""
    assert RawSwitch.op_name == "RawSwitch"
    assert RawMerge.op_name == "RawMerge"
    assert RawConv2D.op_name == "RawConv2D"
    assert RawMatMul.op_name == "RawMatMul"


def test_raw_ops_definitions() -> None:
    """Verify RawOp base class and dynamic control flow raw operation definitions."""
    raw = RawOp()
    assert raw.infer_shape() == ()
    assert raw.infer_shape((1, 2), flag=True) == ()

    assert RawSwitch().op_name == "RawSwitch"
    assert RawMerge().op_name == "RawMerge"
    assert RawConv2D().op_name == "RawConv2D"
    assert RawMatMul().op_name == "RawMatMul"
