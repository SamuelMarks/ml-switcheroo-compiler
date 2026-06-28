from ml_switcheroo_compiler.ops.nn.conv_utils import (
    _calc_same_pad,
    _calc_valid_pad,
    _calculate_conv_transpose_padding,
)
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.config import ConfigContext
import numpy as np


def test_conv_utils_extra():
    # _calc_same_pad (34-39)
    # to hit 's > k - 1', we need stride > kernel - 1
    assert _calc_same_pad(k=2, s=3) == (1, 2)
    assert _calc_same_pad(k=2, s=1) == (1, 0)

    # _calc_valid_pad (72-81 are in _calculate_conv_transpose_padding or _calc_valid_pad?)
    res = _calc_valid_pad(k=2, s=1)
    assert res == (1, 1)

    # _calculate_conv_transpose_padding
    assert _calculate_conv_transpose_padding("VALID", (2,), (1,)) == [(1, 1)]
    assert _calculate_conv_transpose_padding("SAME", (2,), (1,)) == [(1, 0)]

    # sequence input
    assert _calculate_conv_transpose_padding([(1, 1)], (2,), (1,)) == [(1, 1)]


def test_prepare_depthwise_conv_extra():
    from ml_switcheroo_compiler.ops.nn.conv_utils import _prepare_depthwise_conv

    device = Device("cpu")
    # need spatial_dims = 1
    # rhs shape: spatial_dims + (-2=in_channels, -1=channel_multiplier) -> e.g. 1 + 2 = 3
    t1 = Tensor(
        np.ones((1, 5, 2)), TensorConfig((1, 5, 2), "float32", device)
    )  # batch, spatial, channels
    t2 = Tensor(
        np.ones((3, 2, 4)), TensorConfig((3, 2, 4), "float32", device)
    )  # spatial, in_channels, channel_multiplier

    with ConfigContext(eager_mode=True):
        from unittest.mock import patch

        with patch("ml_switcheroo_compiler.ops.shape.reshape") as mock_reshape:
            mock_reshape.return_value = "reshaped"
            # integer dilations
            rhs_reshaped, conf = _prepare_depthwise_conv(
                t1,
                t2,
                1,
                ((0, 1, 2), (0, 1, 2), (0, 1, 2)),
                strides=1,
                lhs_dilation=1,
                rhs_dilation=1,
            )
            assert rhs_reshaped == "reshaped"
            assert conf.lhs_dilation == (1,)
            assert conf.rhs_dilation == (1,)
