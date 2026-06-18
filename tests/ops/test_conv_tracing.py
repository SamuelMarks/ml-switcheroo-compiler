from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.ops.linalg.conv import conv_general_dilated
from ml_switcheroo_compiler.tracing.tracer import _tracer


from ml_switcheroo_compiler.ops.configs import ConvConfig


def test_conv_tracing():
    device = Device(DeviceType.CPU, 0)
    with ConfigContext(eager_mode=False):
        _tracer.start_tracing()
        try:
            lhs = Tensor("dummy1", (1, 3, 4, 4), DType.Float32, device)
            rhs = Tensor("dummy2", (3, 3, 3, 3), DType.Float32, device)
            conv_general_dilated(lhs, rhs, ConvConfig(window_strides=(1, 1), padding="SAME"))
        finally:
            _tracer.stop_tracing()
