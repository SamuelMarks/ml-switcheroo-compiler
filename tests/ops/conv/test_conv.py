# ruff: noqa: E501
from unittest.mock import patch

import numpy as np
import pytest

import ml_switcheroo_compiler.ops.nn.conv1d as conv1d_mod
import ml_switcheroo_compiler.ops.nn.conv2d as conv2d_mod
import ml_switcheroo_compiler.ops.nn.conv3d as conv3d_mod
import ml_switcheroo_compiler.ops.nn.conv_lstm as conv_lstm_mod
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Device, DType, Tensor, TensorConfig
from ml_switcheroo_compiler.ops.nn.clip_grad import clip_grad_norm


def test_clip_grad_coverage():
    config.eager_mode = True
    t1 = Tensor(np.array([1.0, 2.0]), TensorConfig(shape=(2,), dtype=DType("float32"), device=Device("cpu")))
    t2 = Tensor(np.array([-3.0, 4.0]), TensorConfig(shape=(2,), dtype=DType("float32"), device=Device("cpu")))

    res, norm = clip_grad_norm(t1, 1.0)
    assert res is not None

    res, norm = clip_grad_norm([t1, t2], 1.0, norm_type=float("inf"))
    assert len(res) == 2

    res, norm = clip_grad_norm([t1, t2], 1.0, norm_type=1.0)
    assert len(res) == 2

    res, norm = clip_grad_norm([], 1.0)
    assert res == []


def test_conv1d_coverage():
    config.eager_mode = True
    t_in = Tensor(np.ones((1, 3, 2)), TensorConfig(shape=(1, 3, 2), dtype=DType("float32"), device=Device("cpu")))
    t_w = Tensor(np.ones((2, 2, 4)), TensorConfig(shape=(2, 2, 4), dtype=DType("float32"), device=Device("cpu")))
    t_pw = Tensor(np.ones((1, 8, 4)), TensorConfig(shape=(1, 8, 4), dtype=DType("float32"), device=Device("cpu")))

    res = conv1d_mod.conv1d(t_in, t_w)
    assert res is not None

    with patch("ml_switcheroo_compiler.ops.registry.get_op") as mock_get:
        mock_op = mock_get.return_value.return_value
        mock_op.return_value = "transposed"
        assert conv1d_mod.conv1d_transpose(t_in, t_w) == "transposed"

    res = conv1d_mod.depthwise_conv1d(t_in, t_w)
    assert res is not None

    res = conv1d_mod.separable_conv1d(t_in, t_w, t_pw)
    assert res is not None


def test_conv2d_coverage():
    config.eager_mode = True
    t_in = Tensor(np.ones((1, 3, 3, 2)), TensorConfig(shape=(1, 3, 3, 2), dtype=DType("float32"), device=Device("cpu")))
    t_w = Tensor(np.ones((2, 2, 2, 4)), TensorConfig(shape=(2, 2, 2, 4), dtype=DType("float32"), device=Device("cpu")))
    t_pw = Tensor(np.ones((1, 1, 8, 4)), TensorConfig(shape=(1, 1, 8, 4), dtype=DType("float32"), device=Device("cpu")))

    res = conv2d_mod.conv2d(t_in, t_w)
    assert res is not None

    with patch("ml_switcheroo_compiler.ops.registry.get_op") as mock_get:
        mock_op = mock_get.return_value.return_value
        mock_op.return_value = "transposed"
        assert conv2d_mod.conv2d_transpose(t_in, t_w) == "transposed"

    res = conv2d_mod.depthwise_conv2d(t_in, t_w)
    assert res is not None

    res = conv2d_mod.separable_conv2d(t_in, t_w, t_pw)
    assert res is not None


def test_conv3d_coverage():
    config.eager_mode = True
    t_in = Tensor(np.ones((1, 3, 3, 3, 2)), TensorConfig(shape=(1, 3, 3, 3, 2), dtype=DType("float32"), device=Device("cpu")))
    t_w = Tensor(np.ones((2, 2, 2, 2, 4)), TensorConfig(shape=(2, 2, 2, 2, 4), dtype=DType("float32"), device=Device("cpu")))

    res = conv3d_mod.conv3d(t_in, t_w)
    assert res is not None

    with patch("ml_switcheroo_compiler.ops.registry.get_op") as mock_get:
        mock_op = mock_get.return_value.return_value
        mock_op.return_value = "transposed"
        assert conv3d_mod.conv3d_transpose(t_in, t_w) == "transposed"


def test_conv_lstm_coverage():
    config.eager_mode = True
    from ml_switcheroo_compiler.ops.nn.rnn_utils import RNNWeights

    weights_1d = RNNWeights(
        kernel=Tensor(np.ones((2, 2, 8)), TensorConfig(shape=(2, 2, 8), dtype=DType("float32"), device=Device("cpu"))),
        recurrent_kernel=Tensor(np.ones((2, 2, 8)), TensorConfig(shape=(2, 2, 8), dtype=DType("float32"), device=Device("cpu"))),
        bias=Tensor(np.ones((8,)), TensorConfig(shape=(8,), dtype=DType("float32"), device=Device("cpu"))),
    )
    t_in_1d = Tensor(np.ones((1, 3, 2)), TensorConfig(shape=(1, 3, 2), dtype=DType("float32"), device=Device("cpu")))
    state_1d = (Tensor(np.ones((1, 3, 2)), TensorConfig(shape=(1, 3, 2), dtype=DType("float32"), device=Device("cpu"))), Tensor(np.ones((1, 3, 2)), TensorConfig(shape=(1, 3, 2), dtype=DType("float32"), device=Device("cpu"))))

    # Needs valid dimensions to pass the gates, which split features into 4 chunks (so output channels should be multiple of 4)
    # Our weights_1d output channels is 8. And state has channels 2, but Wait, we need it to split properly!

    with patch("ml_switcheroo_compiler.ops.nn.conv_lstm._apply_conv_lstm_gates") as mock_gates:
        mock_gates.return_value = ("hidden", "cell")

        # 1D
        assert conv_lstm_mod.conv_lstm_cell(t_in_1d, state_1d, weights_1d) == ("hidden", "cell")

        # 2D
        weights_2d = RNNWeights(
            kernel=Tensor(np.ones((2, 2, 2, 8)), TensorConfig(shape=(2, 2, 2, 8), dtype=DType("float32"), device=Device("cpu"))),
            recurrent_kernel=Tensor(np.ones((2, 2, 2, 8)), TensorConfig(shape=(2, 2, 2, 8), dtype=DType("float32"), device=Device("cpu"))),
            bias=Tensor(np.ones((8,)), TensorConfig(shape=(8,), dtype=DType("float32"), device=Device("cpu"))),
        )
        t_in_2d = Tensor(np.ones((1, 3, 3, 2)), TensorConfig(shape=(1, 3, 3, 2), dtype=DType("float32"), device=Device("cpu")))
        state_2d = (Tensor(np.ones((1, 3, 3, 2)), TensorConfig(shape=(1, 3, 3, 2), dtype=DType("float32"), device=Device("cpu"))), Tensor(np.ones((1, 3, 3, 2)), TensorConfig(shape=(1, 3, 3, 2), dtype=DType("float32"), device=Device("cpu"))))
        assert conv_lstm_mod.conv_lstm_cell(t_in_2d, state_2d, weights_2d) == ("hidden", "cell")

        # 3D
        weights_3d = RNNWeights(kernel=Tensor(np.ones((2, 2, 2, 2, 8)), TensorConfig(shape=(2, 2, 2, 2, 8), dtype=DType("float32"), device=Device("cpu"))), recurrent_kernel=Tensor(np.ones((2, 2, 2, 2, 8)), TensorConfig(shape=(2, 2, 2, 2, 8), dtype=DType("float32"), device=Device("cpu"))), bias=None)
        t_in_3d = Tensor(np.ones((1, 3, 3, 3, 2)), TensorConfig(shape=(1, 3, 3, 3, 2), dtype=DType("float32"), device=Device("cpu")))
        state_3d = (Tensor(np.ones((1, 3, 3, 3, 2)), TensorConfig(shape=(1, 3, 3, 3, 2), dtype=DType("float32"), device=Device("cpu"))), Tensor(np.ones((1, 3, 3, 3, 2)), TensorConfig(shape=(1, 3, 3, 3, 2), dtype=DType("float32"), device=Device("cpu"))))
        assert conv_lstm_mod.conv_lstm_cell(t_in_3d, state_3d, weights_3d) == ("hidden", "cell")

        # Unsupported
        with pytest.raises(ValueError):
            conv_lstm_mod.conv_lstm_cell(Tensor(np.ones((1, 2)), TensorConfig(shape=(1, 2), dtype=DType("float32"), device=Device("cpu"))), state_1d, weights_1d)

    # Test _apply_conv_lstm_gates
    x_conv = Tensor(np.ones((1, 3, 8)), TensorConfig(shape=(1, 3, 8), dtype=DType("float32"), device=Device("cpu")))
    h_conv = Tensor(np.ones((1, 3, 8)), TensorConfig(shape=(1, 3, 8), dtype=DType("float32"), device=Device("cpu")))
    new_h, state = conv_lstm_mod._apply_conv_lstm_gates(x_conv, h_conv, state_1d, weights_1d, "channels_last")
    assert new_h is not None
