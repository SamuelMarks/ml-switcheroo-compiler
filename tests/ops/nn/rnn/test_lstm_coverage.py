"""Tests for test_lstm_coverage."""

from __future__ import annotations

import numpy as np

import ml_switcheroo_compiler.ops.nn.lstm as lstm_mod
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


def test_lstm_full_coverage() -> None:
    """Test full coverage for LSTM cell and activation helper."""
    inputs = Tensor(np.ones((2, 3), dtype=np.float32), TensorConfig((2, 3), "float32", "cpu"))
    h = Tensor(np.ones((2, 4), dtype=np.float32), TensorConfig((2, 4), "float32", "cpu"))
    c = Tensor(np.ones((2, 4), dtype=np.float32), TensorConfig((2, 4), "float32", "cpu"))
    kernel = Tensor(np.ones((3, 16), dtype=np.float32), TensorConfig((3, 16), "float32", "cpu"))
    rec_kernel = Tensor(np.ones((4, 16), dtype=np.float32), TensorConfig((4, 16), "float32", "cpu"))
    bias = Tensor(np.ones((16,), dtype=np.float32), TensorConfig((16,), "float32", "cpu"))

    original_eager = config.eager_mode
    try:
        config.eager_mode = True
        h_out, (h_new, c_new) = lstm_mod.lstm_cell(inputs, (h, c), kernel, rec_kernel, bias=None)
        assert h_out is not None
        assert h_new.shape == (2, 4)
        assert c_new.shape == (2, 4)

        h_out_bias, (h_new_b, c_new_b) = lstm_mod.lstm_cell(inputs, (h, c), kernel, rec_kernel, bias=bias)
        assert h_out_bias is not None
        assert h_new_b.shape == (2, 4)
        assert c_new_b.shape == (2, 4)

        sig = lstm_mod._sigmoid(inputs)
        assert sig is not None
    finally:
        config.eager_mode = original_eager
