"""Tests for test_conv_lstm_coverage."""

from __future__ import annotations

import importlib
from unittest.mock import patch

import numpy as np
import pytest

import ml_switcheroo_compiler.ops.nn.conv_lstm as conv_lstm_mod
import ml_switcheroo_compiler.ops.nn.rnn_utils as rnn_utils_mod
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

gru_mod = importlib.import_module("ml_switcheroo_compiler.ops.nn.gru")


class DummyWithShape:
    """Mock operand providing a shape attribute."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Initialize mock with specific shape.

        Args:
            shape (tuple[int, ...]): Target shape.
        """
        self.shape: tuple[int, ...] = shape


class DummyWithoutShape:
    """Mock operand lacking a shape attribute."""

    def __init__(self) -> None:
        """Initialize mock without shape."""
        self.val: int = 42


def test_nn_conv_lstm_coverage() -> None:
    """Verify 100% line and branch coverage for conv_lstm operations."""
    orig_eager = config.eager_mode
    try:
        config.eager_mode = True
        # 1D Conv LSTM: mock conv1d to return valid gate outputs (shape with 4 channels/gates at the end)
        mock_gate_1d = Tensor(np.ones((1, 8, 16)), TensorConfig((1, 8, 16), DType.Float32, Device("cpu")))
        x1d = Tensor(np.ones((1, 8, 2)), TensorConfig((1, 8, 2), DType.Float32, Device("cpu")))
        h1d = Tensor(np.ones((1, 8, 4)), TensorConfig((1, 8, 4), DType.Float32, Device("cpu")))
        c1d = Tensor(np.ones((1, 8, 4)), TensorConfig((1, 8, 4), DType.Float32, Device("cpu")))
        w1d = Tensor(np.ones((3, 2, 16)), TensorConfig((3, 2, 16), DType.Float32, Device("cpu")))
        rw1d = Tensor(np.ones((3, 4, 16)), TensorConfig((3, 4, 16), DType.Float32, Device("cpu")))
        b1d = Tensor(np.ones((16,)), TensorConfig((16,), DType.Float32, Device("cpu")))

        weights1d = rnn_utils_mod.RNNWeights(kernel=w1d, recurrent_kernel=rw1d, bias=b1d)
        weights1d_nobias = rnn_utils_mod.RNNWeights(kernel=w1d, recurrent_kernel=rw1d, bias=None)

        with patch("ml_switcheroo_compiler.ops.nn.conv_lstm.conv1d", return_value=mock_gate_1d):
            # Test conv1d_lstm_cell with custom config and default config
            res_h1d, (next_h1d, next_c1d) = conv_lstm_mod.conv1d_lstm_cell(
                x1d,
                (h1d, c1d),
                weights1d,
                config=rnn_utils_mod.ConvLSTMConfig(strides=1, padding="SAME", data_format="channels_last"),
            )
            assert isinstance(res_h1d, Tensor)
            assert isinstance(next_h1d, Tensor)
            assert isinstance(next_c1d, Tensor)

            # Without config (defaults to ConvLSTMConfig()) and without bias
            res_h1d_nb, _ = conv_lstm_mod.conv1d_lstm_cell(x1d, (h1d, c1d), weights1d_nobias, config=None)
            assert isinstance(res_h1d_nb, Tensor)

            # Dispatch via conv_lstm_cell with rank 3
            res_disp1d, _ = conv_lstm_mod.conv_lstm_cell(x1d, (h1d, c1d), weights1d)
            assert isinstance(res_disp1d, Tensor)

        # 2D Conv LSTM
        mock_gate_2d = Tensor(np.ones((1, 4, 4, 16)), TensorConfig((1, 4, 4, 16), DType.Float32, Device("cpu")))
        mock_gate_2d_cf = Tensor(np.ones((1, 16, 4, 4)), TensorConfig((1, 16, 4, 4), DType.Float32, Device("cpu")))
        x2d = Tensor(np.ones((1, 4, 4, 2)), TensorConfig((1, 4, 4, 2), DType.Float32, Device("cpu")))
        h2d = Tensor(np.ones((1, 4, 4, 4)), TensorConfig((1, 4, 4, 4), DType.Float32, Device("cpu")))
        c2d = Tensor(np.ones((1, 4, 4, 4)), TensorConfig((1, 4, 4, 4), DType.Float32, Device("cpu")))
        w2d = Tensor(np.ones((3, 3, 2, 16)), TensorConfig((3, 3, 2, 16), DType.Float32, Device("cpu")))
        rw2d = Tensor(np.ones((3, 3, 4, 16)), TensorConfig((3, 3, 4, 16), DType.Float32, Device("cpu")))
        weights2d = rnn_utils_mod.RNNWeights(kernel=w2d, recurrent_kernel=rw2d, bias=b1d)

        # Test conv2d_lstm_cell with channels_first
        b_cf = Tensor(np.ones((1, 16, 1, 1)), TensorConfig((1, 16, 1, 1), DType.Float32, Device("cpu")))
        weights2d_cf = rnn_utils_mod.RNNWeights(kernel=w2d, recurrent_kernel=rw2d, bias=b_cf)
        with patch("ml_switcheroo_compiler.ops.nn.conv_lstm.conv2d", return_value=mock_gate_2d_cf):
            h2d_cf = Tensor(np.ones((1, 4, 4, 4)), TensorConfig((1, 4, 4, 4), DType.Float32, Device("cpu")))
            c2d_cf = Tensor(np.ones((1, 4, 4, 4)), TensorConfig((1, 4, 4, 4), DType.Float32, Device("cpu")))
            res_h2d_cf, _ = conv_lstm_mod.conv2d_lstm_cell(
                x2d,
                (h2d_cf, c2d_cf),
                weights2d_cf,
                config=rnn_utils_mod.ConvLSTMConfig(data_format="channels_first"),
            )
            assert isinstance(res_h2d_cf, Tensor)

        # conv2d_lstm_cell with config=None (channels_last)
        with patch("ml_switcheroo_compiler.ops.nn.conv_lstm.conv2d", return_value=mock_gate_2d):
            res_h2d, _ = conv_lstm_mod.conv2d_lstm_cell(x2d, (h2d, c2d), weights2d, config=None)
            assert isinstance(res_h2d, Tensor)

            # Dispatch via conv_lstm_cell with rank 4
            res_disp2d, _ = conv_lstm_mod.conv_lstm_cell(x2d, (h2d, c2d), weights2d)
            assert isinstance(res_disp2d, Tensor)

        # 3D Conv LSTM
        mock_gate_3d = Tensor(np.ones((1, 2, 2, 2, 16)), TensorConfig((1, 2, 2, 2, 16), DType.Float32, Device("cpu")))
        x3d = Tensor(np.ones((1, 2, 2, 2, 2)), TensorConfig((1, 2, 2, 2, 2), DType.Float32, Device("cpu")))
        h3d = Tensor(np.ones((1, 2, 2, 2, 4)), TensorConfig((1, 2, 2, 2, 4), DType.Float32, Device("cpu")))
        c3d = Tensor(np.ones((1, 2, 2, 2, 4)), TensorConfig((1, 2, 2, 2, 4), DType.Float32, Device("cpu")))
        w3d = Tensor(np.ones((1, 1, 1, 2, 16)), TensorConfig((1, 1, 1, 2, 16), DType.Float32, Device("cpu")))
        rw3d = Tensor(np.ones((1, 1, 1, 4, 16)), TensorConfig((1, 1, 1, 4, 16), DType.Float32, Device("cpu")))
        weights3d = rnn_utils_mod.RNNWeights(kernel=w3d, recurrent_kernel=rw3d, bias=b1d)

        with patch("ml_switcheroo_compiler.ops.nn.conv_lstm.conv3d", return_value=mock_gate_3d):
            # conv3d_lstm_cell with config=None
            res_h3d, _ = conv_lstm_mod.conv3d_lstm_cell(x3d, (h3d, c3d), weights3d, config=None)
            assert isinstance(res_h3d, Tensor)

            # conv3d_lstm_cell with custom config
            res_h3d_conf, _ = conv_lstm_mod.conv3d_lstm_cell(
                x3d,
                (h3d, c3d),
                weights3d,
                config=rnn_utils_mod.ConvLSTMConfig(strides=1, padding="SAME", data_format="channels_last"),
            )
            assert isinstance(res_h3d_conf, Tensor)

            # Dispatch via conv_lstm_cell with rank 5
            res_disp3d, _ = conv_lstm_mod.conv_lstm_cell(x3d, (h3d, c3d), weights3d)
            assert isinstance(res_disp3d, Tensor)

        # Unsupported dimension (e.g. rank 2)
        x_bad = Tensor(np.ones((1, 2)), TensorConfig((1, 2), DType.Float32, Device("cpu")))
        with pytest.raises(ValueError, match="Unsupported input dimension for conv_lstm_cell"):
            conv_lstm_mod.conv_lstm_cell(x_bad, (h1d, c1d), weights1d)

        # Directly test conv_lstm _sigmoid
        sig_out = conv_lstm_mod._sigmoid(x1d)
        assert sig_out is not None
    finally:
        config.eager_mode = orig_eager
