"""Test Numpy eager variable ops coverage."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.numpy.eager.variable_ops import _np_assign_variable
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol


def test_assign_variable_op() -> None:
    """Test function test_assign_variable_op."""
    assert _np_assign_variable(typing.cast("BackendModuleProtocol", None), "dummy_ref", "real_value") == "real_value"
    assert _np_assign_variable(typing.cast("BackendModuleProtocol", None), "dummy_ref") is None
