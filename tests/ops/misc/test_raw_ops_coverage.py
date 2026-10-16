"""Tests for test_raw_ops_coverage."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ml_switcheroo_compiler.ops.raw_ops import RawConv2D, RawMatMul, RawMerge, RawOp, RawSwitch

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_raw_ops_definitions() -> None:
    """Verify RawOp base class and dynamic control flow raw operation definitions."""
    raw = RawOp()
    assert raw.infer_shape() == ()
    assert cast("Callable[..., EagerTensorProtocol]", raw.infer_shape)((1, 2), flag=True) == ()

    assert RawSwitch().op_name == "RawSwitch"
    assert RawMerge().op_name == "RawMerge"
    assert RawConv2D().op_name == "RawConv2D"
    assert RawMatMul().op_name == "RawMatMul"
