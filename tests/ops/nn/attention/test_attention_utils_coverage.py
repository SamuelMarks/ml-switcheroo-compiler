"""Tests for test_attention_utils_coverage."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

import ml_switcheroo_compiler.ops.nn.attention_utils as attn_utils_mod
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig


class DummyWithShape:
    """Mock operand providing shape attribute."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Initialize mock with specific shape.

        Args:
            shape (tuple[int, ...]): Target shape.
        """
        self.shape: tuple[int, ...] = shape


class DummyWithoutShape:
    """Mock operand lacking shape attribute."""

    def __init__(self) -> None:
        """Initialize mock without shape."""
        self.val: int = 123


def test_attention_utils_full_coverage() -> None:
    """Test full coverage for attention mechanism utilities."""
    rope_op = attn_utils_mod.RopeOp()
    assert rope_op.infer_shape(DummyWithShape((2, 4, 8))) == (2, 4, 8)
    assert rope_op.infer_shape(DummyWithoutShape()) == ()

    t_in = Tensor(np.ones((2, 4, 8), dtype=np.float32), TensorConfig((2, 4, 8), "float32", "cpu"))

    original_eager = config.eager_mode
    try:
        config.eager_mode = True
        with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_be:
            mock_be.return_value.execute_op.return_value = np.ones((2, 4, 8), dtype=np.float32)
            res_eager = attn_utils_mod.rope(t_in, axis=8)
            assert isinstance(res_eager, Tensor)

        config.eager_mode = False
        with patch("ml_switcheroo_compiler.ops.nn.attention_utils._emit_shape_node", return_value=t_in) as mock_emit:
            res_symbolic = attn_utils_mod.rope(t_in, axis=8)
            assert res_symbolic is t_in
            mock_emit.assert_called_once()
    finally:
        config.eager_mode = original_eager

    original_eager = config.eager_mode
    try:
        config.eager_mode = True
        pe = attn_utils_mod.sinusoidal_positional_encoding(seq_len=4, axis=8)
        assert isinstance(pe, Tensor)

        mask = attn_utils_mod.alibi_mask(seq_len=4, num_heads=2)
        assert mask == 0.0
    finally:
        config.eager_mode = original_eager
