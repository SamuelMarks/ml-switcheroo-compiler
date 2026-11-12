"""Tests for test_attention_rope_alibi."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.nn.attention import ScaledDotProductAttention
from ml_switcheroo_compiler.ops.nn.attention_utils import RopeOp, alibi_mask, rope, sinusoidal_positional_encoding
from tests.testing_utils import ensure_callable, ensure_eager_tensor, ensure_mutable

if typing.TYPE_CHECKING:
    pass


def test_rope_infer_shape() -> None:
    """Test function test_rope_infer_shape."""

    class DummyShape:
        """Test class DummyShape."""

        shape = (1, 2)

    assert ensure_callable(RopeOp)().infer_shape(ensure_callable(DummyShape)()) == (1, 2)


def test_rope() -> None:
    """Test function test_rope."""
    t = Tensor(np.array([[1.0, 2.0]]), TensorConfig(shape=(1, 2), dtype=DType("float32"), device=Device("cpu")))
    orig = config.eager_mode
    config.eager_mode = True
    eager_out = rope(t, 2)
    assert isinstance(eager_out, Tensor) and eager_out.shape == (1, 2)
    config.eager_mode = False
    import ml_switcheroo_compiler.tracing.state as state

    orig_tracing = state.global_tracing_state.is_tracing
    state.global_tracing_state.is_tracing = True
    orig_add_node = state.global_tracing_state.add_node
    added_nodes: list[str] = []
    ensure_mutable(state.global_tracing_state).add_node = ensure_eager_tensor(lambda node: added_nodes.append(str(ensure_eager_tensor(node).op_type)))
    try:
        traced_out = rope(t, 2)
        assert isinstance(traced_out, Tensor) and traced_out.shape == (1, 2)
        assert len(added_nodes) >= 1
    finally:
        config.eager_mode = orig
        state.global_tracing_state.is_tracing = orig_tracing
        ensure_mutable(state.global_tracing_state).add_node = ensure_eager_tensor(orig_add_node)


def test_sinusoidal() -> None:
    """Test function test_sinusoidal."""
    orig = config.eager_mode
    config.eager_mode = False
    import ml_switcheroo_compiler.tracing.state as state

    orig_tracing = state.global_tracing_state.is_tracing
    state.global_tracing_state.is_tracing = True
    orig_add_node = state.global_tracing_state.add_node
    ensure_mutable(state.global_tracing_state).add_node = ensure_eager_tensor(lambda node: None)
    try:
        res = sinusoidal_positional_encoding(4, 4)
        assert res is not None
    finally:
        config.eager_mode = orig
        state.global_tracing_state.is_tracing = orig_tracing
        ensure_mutable(state.global_tracing_state).add_node = ensure_eager_tensor(orig_add_node)


def test_alibi() -> None:
    """Test function test_alibi."""
    orig = config.eager_mode
    config.eager_mode = False
    import ml_switcheroo_compiler.tracing.state as state

    orig_tracing = state.global_tracing_state.is_tracing
    state.global_tracing_state.is_tracing = True
    orig_add_node = state.global_tracing_state.add_node
    ensure_mutable(state.global_tracing_state).add_node = ensure_eager_tensor(lambda node: None)
    try:
        res = alibi_mask(4, 2)
        assert res == 0.0
    finally:
        config.eager_mode = orig
        state.global_tracing_state.is_tracing = orig_tracing
        ensure_mutable(state.global_tracing_state).add_node = ensure_eager_tensor(orig_add_node)


def test_scaled_dot() -> None:
    """Test function test_scaled_dot."""

    class DummyShape:
        """Test class DummyShape."""

        shape = (1, 2)

    assert ensure_callable(ScaledDotProductAttention)().infer_shape(ensure_callable(DummyShape)(), None, None, None) == (1, 2)
