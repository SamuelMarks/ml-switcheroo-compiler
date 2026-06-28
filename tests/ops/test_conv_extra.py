from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.linalg.conv import conv_general_dilated_local
from ml_switcheroo_compiler.tracing.tracer import _tracer


def test_conv_general_dilated_local_coverage():
    device = Device(DeviceType.CPU, 0)
    with ConfigContext(eager_mode=False):
        _tracer.start_tracing()
        try:
            t1 = Tensor("dummy1", TensorConfig((1, 1, 5, 5), DType.Float32, device))
            t2 = Tensor("dummy2", TensorConfig((1, 1, 3, 3), DType.Float32, device))
            conv_general_dilated_local(t1, t2, (1, 1), "SAME", (3, 3))
        finally:
            _tracer.stop_tracing()
