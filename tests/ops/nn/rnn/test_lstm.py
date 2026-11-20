"""Behavioral unit tests for lstm."""

from __future__ import annotations

import numpy as np

import ml_switcheroo_compiler.ops.nn.lstm as lstm_mod
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Device, DType, Tensor, TensorConfig
from ml_switcheroo_compiler.ops.nn.lstm import lstm_cell


def test_lstm_coverage() -> None:
    """Test function test_lstm_coverage."""
    config.eager_mode = True
    t_in = Tensor(np.array([[1.0, 2.0]]), TensorConfig(shape=(1, 2), dtype=DType("float32"), device=Device("cpu")))
    t_h = Tensor(np.array([[1.0, 1.0]]), TensorConfig(shape=(1, 2), dtype=DType("float32"), device=Device("cpu")))
    t_c = Tensor(np.array([[1.0, 1.0]]), TensorConfig(shape=(1, 2), dtype=DType("float32"), device=Device("cpu")))
    w = Tensor(np.array([[1.0] * 8] * 2), TensorConfig(shape=(2, 8), dtype=DType("float32"), device=Device("cpu")))
    rw = Tensor(np.array([[1.0] * 8] * 2), TensorConfig(shape=(2, 8), dtype=DType("float32"), device=Device("cpu")))
    b = Tensor(np.array([1.0] * 8), TensorConfig(shape=(8,), dtype=DType("float32"), device=Device("cpu")))

    res_h, (res_h_next, res_c_next) = lstm_cell(t_in, (t_h, t_c), w, rw, b)
    assert res_h.shape == (1, 2)
    assert res_h_next.shape == (1, 2)
    assert res_c_next.shape == (1, 2)
    res_h2, (res_h2_next, res_c2_next) = lstm_cell(t_in, (t_h, t_c), w, rw)
    assert res_h2.shape == (1, 2)
    assert res_h2_next.shape == (1, 2)
    assert res_c2_next.shape == (1, 2)


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
