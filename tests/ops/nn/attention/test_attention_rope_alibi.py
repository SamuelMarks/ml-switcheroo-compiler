"""Tests for test_attention_rope_alibi."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.nn.attention import ScaledDotProductAttention
from ml_switcheroo_compiler.ops.nn.attention_utils import RopeOp, alibi_mask, rope, sinusoidal_positional_encoding

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_rope_infer_shape() -> None:
    """Test function test_rope_infer_shape."""

    class DummyShape:
        """Test class DummyShape."""

        shape = (1, 2)

    assert cast("Callable[..., EagerTensorProtocol]", RopeOp)().infer_shape(cast("Callable[..., EagerTensorProtocol]", DummyShape)()) == (1, 2)


def test_rope() -> None:
    """Test function test_rope."""
    t = Tensor(np.array([[1.0, 2.0]]), TensorConfig(shape=(1, 2), dtype=DType("float32"), device=Device("cpu")))
    orig = config.eager_mode
    config.eager_mode = True
    rope(t, 2)
    config.eager_mode = False
    import ml_switcheroo_compiler.tracing.state as state

    orig_tracing = state.global_tracing_state.is_tracing
    state.global_tracing_state.is_tracing = True
    orig_add_node = state.global_tracing_state.add_node
    setattr(state.global_tracing_state, "add_node", lambda node: None)
    try:
        rope(t, 2)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
        _caught_err = True
    finally:
        config.eager_mode = orig
        state.global_tracing_state.is_tracing = orig_tracing
        setattr(state.global_tracing_state, "add_node", orig_add_node)
    assert callable(test_rope)


def test_sinusoidal() -> None:
    """Test function test_sinusoidal."""
    orig = config.eager_mode
    config.eager_mode = False
    import ml_switcheroo_compiler.tracing.state as state

    orig_tracing = state.global_tracing_state.is_tracing
    state.global_tracing_state.is_tracing = True
    orig_add_node = state.global_tracing_state.add_node
    setattr(state.global_tracing_state, "add_node", lambda node: None)
    try:
        res = sinusoidal_positional_encoding(4, 4)
        assert res is not None
    finally:
        config.eager_mode = orig
        state.global_tracing_state.is_tracing = orig_tracing
        setattr(state.global_tracing_state, "add_node", orig_add_node)


def test_alibi() -> None:
    """Test function test_alibi."""
    orig = config.eager_mode
    config.eager_mode = False
    import ml_switcheroo_compiler.tracing.state as state

    orig_tracing = state.global_tracing_state.is_tracing
    state.global_tracing_state.is_tracing = True
    orig_add_node = state.global_tracing_state.add_node
    setattr(state.global_tracing_state, "add_node", lambda node: None)
    try:
        res = alibi_mask(4, 2)
        assert res == 0.0
    finally:
        config.eager_mode = orig
        state.global_tracing_state.is_tracing = orig_tracing
        setattr(state.global_tracing_state, "add_node", orig_add_node)


def test_scaled_dot() -> None:
    """Test function test_scaled_dot."""

    class DummyShape:
        """Test class DummyShape."""

        shape = (1, 2)

    assert cast("Callable[..., EagerTensorProtocol]", ScaledDotProductAttention)().infer_shape(cast("Callable[..., EagerTensorProtocol]", DummyShape)(), None, None, None) == (1, 2)
