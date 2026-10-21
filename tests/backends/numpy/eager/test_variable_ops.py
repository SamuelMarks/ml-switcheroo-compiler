"""Test Numpy eager variable ops coverage."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.numpy.eager.variable_ops import _np_assign_variable
from tests.testing_utils import as_mod


def test_assign_variable_op() -> None:
    """Test function test_assign_variable_op."""
    assert _np_assign_variable(as_mod(None), "dummy_ref", "real_value") == "real_value"
    assert _np_assign_variable(as_mod(None), "dummy_ref") is None
