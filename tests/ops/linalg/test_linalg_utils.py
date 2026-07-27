"""Test linalg utils."""

import pytest

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.linalg.utils import _emit_linalg_node
from ml_switcheroo_compiler.tracing import global_tracing_state


def test_emit_linalg_node_outside_tracing():
    """Test emit_linalg_node outside tracing context."""
    global_tracing_state.is_tracing = False
    inputs = [Tensor(None, TensorConfig((1,), "float32", "cpu"))]
    with pytest.raises(RuntimeError, match="Cannot emit Foo node outside of a tracing context."):
        _emit_linalg_node("Foo", inputs, {}, [[1]], [DType.Float32])
