"""Tests for `ml_switcheroo_compiler.ops.reductions` package initialization and fallback behavior."""

from __future__ import annotations


def test_reductions_init() -> None:
    """Verify standard reduction operator exports are initialized."""
    import ml_switcheroo_compiler.ops.reductions as red

    assert red.sum is not None
    assert red.sum.op_type == "Sum"


def test_reductions_init_missing() -> None:
    """Verify `_safe_get_op` constructs a fallback `OpDef` when `get_op` raises `KeyError`."""
    import importlib
    from unittest.mock import patch

    import ml_switcheroo_compiler.ops.reductions
    from ml_switcheroo_compiler.ops.base import OpDef

    with patch("ml_switcheroo_compiler.ops.base.get_op", side_effect=KeyError("mock")):
        importlib.reload(ml_switcheroo_compiler.ops.reductions)
        fallback_sum = ml_switcheroo_compiler.ops.reductions.sum
        assert isinstance(fallback_sum, OpDef)
        assert fallback_sum.op_type == "Sum"

    importlib.reload(ml_switcheroo_compiler.ops.reductions)
