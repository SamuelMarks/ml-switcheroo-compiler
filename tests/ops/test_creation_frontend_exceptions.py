"""Provides required module functionality."""

import pytest

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.ops.creation.frontend import _emit_constant_node, _emit_creation_node


def test_emit_creation_node_outside_tracing() -> None:
    """Execute the requested function."""
    with pytest.raises(RuntimeError, match="Cannot emit TestOp node outside of a tracing context"):
        _emit_creation_node("TestOp", (2, 2), DType.Float32)


def test_emit_constant_node_outside_tracing() -> None:
    """Execute the requested function."""
    with pytest.raises(
        RuntimeError,
        match="Cannot emit Constant node outside of a tracing context",
    ):
        _emit_constant_node(1.0, DType.Float32)
