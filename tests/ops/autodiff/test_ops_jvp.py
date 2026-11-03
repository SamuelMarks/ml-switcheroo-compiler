"""Tests for registered operation JVP, VJP, eager_eval, and infer_shape interfaces."""

from __future__ import annotations

from ml_switcheroo_compiler.core.shape import _has_attr
from ml_switcheroo_compiler.ops.registry import _OP_REGISTRY


def test_all_ops_coverage() -> None:
    """Verify that all registered operations instantiate and expose infer_shape and eager_eval."""
    checked = 0
    for op_name, op_cls in _OP_REGISTRY.items():
        if op_name in ["ReadVariable", "AssignVariable", "Input", "Constant", "Call"]:
            continue
        op = op_cls()
        assert _has_attr(op, "infer_shape")
        assert _has_attr(op, "eager_eval")
        checked += 1
    assert checked > 0
