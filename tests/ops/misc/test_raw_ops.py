"""Tests for test_raw_ops."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.ops.raw_ops import RawConv2D, RawMatMul, RawMerge, RawOp, RawSwitch

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol


def test_raw_op_infer_shape() -> None:
    """Test function test_raw_op_infer_shape."""
    op = RawOp()
    assert op.infer_shape() == ()
    assert typing.cast("EagerCallableProtocol", op.infer_shape)(1, 2, a=3) == ()


def test_raw_op_registry() -> None:
    """Test function test_raw_op_registry."""
    assert RawSwitch.op_name == "RawSwitch"
    assert RawMerge.op_name == "RawMerge"
    assert RawConv2D.op_name == "RawConv2D"
    assert RawMatMul.op_name == "RawMatMul"
