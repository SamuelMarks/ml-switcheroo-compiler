# ruff: noqa: E501
import sys
from unittest.mock import patch

import numpy as np
import pytest

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Device, DType, Tensor, TensorConfig
from ml_switcheroo_compiler.ops.nn.conv_nd import conv, conv_transpose, depthwise_conv, separable_conv

gru_mod = sys.modules["ml_switcheroo_compiler.ops.nn.gru"]


def test_conv_nd_coverage():
    config.eager_mode = True
    t_1d = Tensor(np.ones((1, 3, 2)), TensorConfig(shape=(1, 3, 2), dtype=DType("float32"), device=Device("cpu")))
    w_1d = Tensor(np.ones((2, 2, 4)), TensorConfig(shape=(2, 2, 4), dtype=DType("float32"), device=Device("cpu")))
    t_2d = Tensor(np.ones((1, 3, 3, 2)), TensorConfig(shape=(1, 3, 3, 2), dtype=DType("float32"), device=Device("cpu")))
    w_2d = Tensor(np.ones((2, 2, 2, 4)), TensorConfig(shape=(2, 2, 2, 4), dtype=DType("float32"), device=Device("cpu")))
    t_3d = Tensor(np.ones((1, 3, 3, 3, 2)), TensorConfig(shape=(1, 3, 3, 3, 2), dtype=DType("float32"), device=Device("cpu")))
    w_3d = Tensor(np.ones((2, 2, 2, 2, 4)), TensorConfig(shape=(2, 2, 2, 2, 4), dtype=DType("float32"), device=Device("cpu")))

    with patch("ml_switcheroo_compiler.ops.nn.conv_nd.get_active_backend") as mock_backend:
        mock_backend.return_value.execute_op.return_value = np.zeros((1, 3, 2))
        res = conv_transpose(t_1d, w_1d)
        assert res is not None

        config.eager_mode = False
        from ml_switcheroo_compiler.tracing.state import global_tracing_state

        global_tracing_state.is_tracing = True

        def dummy_emit(*args, **kwargs):
            return "emitted"

        import ml_switcheroo_compiler.ops.nn.conv_nd as conv_nd_mod

        conv_nd_mod._emit_linalg_node = dummy_emit
        assert conv_transpose(t_1d, w_1d) == "emitted"
        config.eager_mode = True
        global_tracing_state.is_tracing = False

    with patch("ml_switcheroo_compiler.ops.nn.conv_nd.conv1d") as mock_c1:
        mock_c1.return_value = "c1"
        assert conv(t_1d, w_1d) == "c1"

    with patch("ml_switcheroo_compiler.ops.nn.conv_nd.conv2d") as mock_c2:
        mock_c2.return_value = "c2"
        assert conv(t_2d, w_2d) == "c2"

    with patch("ml_switcheroo_compiler.ops.nn.conv_nd.conv3d") as mock_c3:
        mock_c3.return_value = "c3"
        assert conv(t_3d, w_3d) == "c3"

    with pytest.raises(ValueError):
        conv(Tensor(np.ones((1, 2)), TensorConfig(shape=(1, 2), dtype=DType("float32"), device=Device("cpu"))), Tensor(np.ones((1, 2)), TensorConfig(shape=(1, 2), dtype=DType("float32"), device=Device("cpu"))))

    with patch("ml_switcheroo_compiler.ops.nn.conv_nd.depthwise_conv1d") as mock_d1:
        mock_d1.return_value = "d1"
        assert depthwise_conv(t_1d, w_1d) == "d1"

    with patch("ml_switcheroo_compiler.ops.nn.conv_nd.depthwise_conv2d") as mock_d2:
        mock_d2.return_value = "d2"
        assert depthwise_conv(t_2d, w_2d) == "d2"

    with pytest.raises(ValueError):
        depthwise_conv(t_3d, w_3d)

    with patch("ml_switcheroo_compiler.ops.nn.conv_nd.separable_conv1d") as mock_s1:
        mock_s1.return_value = "s1"
        assert separable_conv(t_1d, w_1d, w_1d) == "s1"

    with patch("ml_switcheroo_compiler.ops.nn.conv_nd.separable_conv2d") as mock_s2:
        mock_s2.return_value = "s2"
        assert separable_conv(t_2d, w_2d, w_2d) == "s2"

    with pytest.raises(ValueError):
        separable_conv(t_3d, w_3d, w_3d)
