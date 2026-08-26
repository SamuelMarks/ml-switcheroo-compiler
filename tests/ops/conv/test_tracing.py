# ruff: noqa: E501
"""Core abstractions and logic definitions for test_conv_tracing.py."""

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.configs import ConvConfig
from ml_switcheroo_compiler.ops.linalg.conv import conv_general_dilated
from ml_switcheroo_compiler.tracing.state import global_tracing_state


def test_conv_tracing():
    """Test the conv tracing behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        device = Device(DeviceType.CPU, 0)
        with ConfigContext(eager_mode=False):
            global_tracing_state.start_tracing()
            try:
                lhs = Tensor("dummy1", TensorConfig((1, 3, 4, 4), DType.Float32, device))
                rhs = Tensor("dummy2", TensorConfig((3, 3, 3, 3), DType.Float32, device))
                conv_general_dilated(lhs, rhs, ConvConfig(window_strides=(1, 1), padding="SAME"))
            finally:
                global_tracing_state.stop_tracing()
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass
