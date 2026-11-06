"""Tests for test_init."""

from __future__ import annotations

from tests.testing_utils import ensure_eager_tensor


def test_reductions_init() -> None:
    """Test function test_reductions_init."""
    import ml_switcheroo_compiler.ops.reductions as red

    assert red.sum is not None


def test_reductions_init_missing() -> None:
    """Test function test_reductions_init_missing."""
    import importlib
    from unittest.mock import patch

    import ml_switcheroo_compiler.ops.reductions

    with patch("ml_switcheroo_compiler.ops.base.get_op", side_effect=KeyError("mock")):
        importlib.reload(ml_switcheroo_compiler.ops.reductions)
        assert ensure_eager_tensor(ml_switcheroo_compiler.ops.reductions).sum is None

    importlib.reload(ml_switcheroo_compiler.ops.reductions)
