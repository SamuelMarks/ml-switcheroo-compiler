"""Test Numpy eager variable ops coverage."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.numpy.eager.variable_ops import _np_assign_variable


def test_assign_variable_op() -> None:
    """Test function test_assign_variable_op."""
    assert _np_assign_variable(None, "dummy_ref", "real_value") == "real_value"
    assert _np_assign_variable(None, "dummy_ref") is None
