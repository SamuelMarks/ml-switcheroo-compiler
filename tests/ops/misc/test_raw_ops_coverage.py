"""Tests for test_raw_ops_coverage."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.ops.raw_ops import RawConv2D, RawMatMul, RawMerge, RawOp, RawSwitch

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol


def test_raw_ops_definitions() -> None:
    """Verify RawOp base class and dynamic control flow raw operation definitions."""
    raw = RawOp()
    assert raw.infer_shape() == ()
    assert typing.cast("EagerCallableProtocol", raw.infer_shape)((1, 2), flag=True) == ()

    assert RawSwitch().op_name == "RawSwitch"
    assert RawMerge().op_name == "RawMerge"
    assert RawConv2D().op_name == "RawConv2D"
    assert RawMatMul().op_name == "RawMatMul"
