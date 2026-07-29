import numpy as np

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.nn.conv_utils import _build_conv_config, _prepare_depthwise_conv


def create_eager_tensor(data):
    return Tensor(data, TensorConfig(data.shape, DType.Float32, Device("cpu")))


def test_conv_utils_missing_branches():
    dim_nums = ((0, 3, 1, 2), (3, 2, 0, 1), (0, 3, 1, 2))

    cfg = _build_conv_config({"strides": (1, 1), "lhs_dilation": (1, 1), "rhs_dilation": (1, 1)}, dim_nums)
    assert getattr(cfg, "window_strides", None) == (1, 1)

    with ConfigContext(eager_mode=True):
        lhs = create_eager_tensor(np.ones((1, 2, 2, 3)))
        rhs = create_eager_tensor(np.ones((2, 2, 3, 1)))

        # Test shape validations or config creation
        r, c = _prepare_depthwise_conv(lhs, rhs, 2, dim_nums, config_obj="mock_config")
        assert r.shape == (2, 2, 1, 3)
        assert c == "mock_config"

        r2, c2 = _prepare_depthwise_conv(lhs, rhs, 2, dim_nums, strides=(1, 1), lhs_dilation=(1, 1), rhs_dilation=(1, 1))
        assert r2.shape == (2, 2, 1, 3)
        assert getattr(c2, "window_strides", None) == (1, 1)
