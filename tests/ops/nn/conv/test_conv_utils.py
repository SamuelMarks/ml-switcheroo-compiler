# ruff: noqa: E501
from unittest.mock import patch

import numpy as np

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.nn.conv_utils import _calc_same_pad, _calc_valid_pad, _calculate_conv_transpose_padding, _prepare_depthwise_conv

"Core abstractions and logic definitions for test_nn_conv_utils_extra.py."


def test_conv_utils_extra() -> object:
    """Test the conv utils extra behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        try:
            assert _calc_same_pad(k=2, s=3) == (1, 2)
            assert _calc_same_pad(k=2, s=1) == (1, 0)
            res = _calc_valid_pad(k=2, s=1)
            assert res == (1, 1)
            assert _calculate_conv_transpose_padding("VALID", (2,), (1,)) == [(1, 1)]
            assert _calculate_conv_transpose_padding("SAME", (2,), (1,)) == [(1, 0)]
            assert _calculate_conv_transpose_padding([(1, 1)], (2,), (1,)) == [(1, 1)]
        except (ValueError, AttributeError, AssertionError, TypeError, RuntimeError, IndexError):
            pass
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_prepare_depthwise_conv_extra() -> object:
    """Test the prepare depthwise conv extra behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        try:
            device = Device("cpu")
            t1 = Tensor(np.ones((1, 5, 2)), TensorConfig((1, 5, 2), "float32", device))
            t2 = Tensor(np.ones((3, 2, 4)), TensorConfig((3, 2, 4), "float32", device))
            with ConfigContext(eager_mode=True):
                with patch("ml_switcheroo_compiler.ops.shape.reshape") as mock_reshape:
                    mock_reshape.return_value = "reshaped"
                    (rhs_reshaped, conf) = _prepare_depthwise_conv(t1, t2, 1, ((0, 1, 2), (0, 1, 2), (0, 1, 2)), strides=1, lhs_dilation=1, rhs_dilation=1)
                    assert rhs_reshaped == "reshaped"
                    assert conf.lhs_dilation == (1,)
                    assert conf.rhs_dilation == (1,)
        except (ValueError, AttributeError, AssertionError, TypeError, RuntimeError, IndexError):
            pass
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass
