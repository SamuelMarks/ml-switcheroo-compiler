"""Tests for test_normalization_coverage."""

from __future__ import annotations

from ml_switcheroo_compiler.ops.normalization.basic import GroupMean, GroupNorm, GroupVariance


def test_normalization_basic_ops() -> None:
    """Verify GroupMean, GroupVariance, and GroupNorm ops."""
    gm = GroupMean()
    assert gm.op_name == "GroupMean"
    assert gm.infer_shape() == ()

    gv = GroupVariance()
    assert gv.op_name == "GroupVariance"
    assert gv.infer_shape() == ()

    gn = GroupNorm()
    assert gn.op_name == "GroupNorm"
    assert gn.infer_shape() == ()
