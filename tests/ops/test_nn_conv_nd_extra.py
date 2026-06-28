from ml_switcheroo_compiler.ops.nn.conv_nd import (
    conv,
    depthwise_conv,
    separable_conv,
    GenericConvConfig,
)
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.config import ConfigContext
import numpy as np


def test_conv_nd_extra():
    device = Device("cpu")
    # For conv
    t1_1d = Tensor(np.ones((1, 2, 5)), TensorConfig((1, 2, 5), "float32", device))
    t2_1d = Tensor(np.ones((3, 2, 3)), TensorConfig((3, 2, 3), "float32", device))
    conf_1d = GenericConvConfig(strides=1, padding="VALID", dilation_rate=1)

    with ConfigContext(eager_mode=True):
        from unittest.mock import patch

        with patch("ml_switcheroo_compiler.ops.nn.conv_nd.conv1d") as mock_conv1d:
            mock_conv1d.return_value = "res"
            conv(t1_1d, t2_1d, conf_1d)
            assert mock_conv1d.called

        # For depthwise_conv 1d
        t1_depth_1d = Tensor(np.ones((1, 2, 5)), TensorConfig((1, 2, 5), "float32", device))
        t2_depth_1d = Tensor(np.ones((2, 1, 3)), TensorConfig((2, 1, 3), "float32", device))
        with patch("ml_switcheroo_compiler.ops.nn.conv_nd.depthwise_conv1d") as mock_dw:
            mock_dw.return_value = "res"
            depthwise_conv(t1_depth_1d, t2_depth_1d, conf_1d)
            assert mock_dw.called

        # For separable_conv 1d
        t1_sep_1d = Tensor(np.ones((1, 2, 5)), TensorConfig((1, 2, 5), "float32", device))
        t2_sep_dw = Tensor(np.ones((2, 1, 3)), TensorConfig((2, 1, 3), "float32", device))
        t2_sep_pw = Tensor(np.ones((4, 2, 1)), TensorConfig((4, 2, 1), "float32", device))
        with patch("ml_switcheroo_compiler.ops.nn.conv_nd.separable_conv1d") as mock_sep:
            mock_sep.return_value = "res"
            separable_conv(t1_sep_1d, t2_sep_dw, t2_sep_pw, conf_1d)
            assert mock_sep.called


def test_conv_transpose_extra():
    from ml_switcheroo_compiler.ops.nn.conv_nd import conv_transpose

    device = Device("cpu")
    t1 = Tensor(np.ones((1, 2, 5)), TensorConfig((1, 2, 5), "float32", device))
    t2 = Tensor(np.ones((3, 2, 3)), TensorConfig((3, 2, 3), "float32", device))

    with ConfigContext(eager_mode=True):
        from unittest.mock import patch

        with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
            mock_backend.return_value.execute_op.return_value = np.ones((1, 3, 7))
            res = conv_transpose(t1, t2, strides=1, padding="VALID")
            assert res is not None

    with ConfigContext(eager_mode=False):
        with patch("ml_switcheroo_compiler.ops.linalg.frontend._emit_linalg_node") as mock_emit:
            conv_transpose(t1, t2, strides=1, padding="VALID")
            assert mock_emit.called


def test_conv_nd_2d_3d():
    device = Device("cpu")
    t1_2d = Tensor(np.ones((1, 2, 5, 5)), TensorConfig((1, 2, 5, 5), "float32", device))
    t2_2d = Tensor(np.ones((3, 2, 3, 3)), TensorConfig((3, 2, 3, 3), "float32", device))
    t1_3d = Tensor(np.ones((1, 2, 5, 5, 5)), TensorConfig((1, 2, 5, 5, 5), "float32", device))
    t2_3d = Tensor(np.ones((3, 2, 3, 3, 3)), TensorConfig((3, 2, 3, 3, 3), "float32", device))
    conf = GenericConvConfig(strides=1, padding="VALID", dilation_rate=1)

    with ConfigContext(eager_mode=True):
        from unittest.mock import patch

        with patch("ml_switcheroo_compiler.ops.nn.conv_nd.conv2d") as mock_conv2d:
            mock_conv2d.return_value = "res"
            conv(t1_2d, t2_2d, conf)
            assert mock_conv2d.called

        with patch("ml_switcheroo_compiler.ops.nn.conv_nd.conv3d") as mock_conv3d:
            mock_conv3d.return_value = "res"
            conv(t1_3d, t2_3d, conf)
            assert mock_conv3d.called

        # depthwise conv error on > 2d
        import pytest

        with pytest.raises(ValueError):
            depthwise_conv(t1_3d, t2_3d, conf)

        # separable conv error on > 2d
        with pytest.raises(ValueError):
            separable_conv(t1_3d, t2_3d, t2_3d, conf)
