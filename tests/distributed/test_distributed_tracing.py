from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.distributed import pbroadcast, pdot, ppermute, pshuffle
from ml_switcheroo_compiler.tracing.tracer import _tracer


def test_distributed_tracing():
    device = Device(DeviceType.CPU, 0)
    with ConfigContext(eager_mode=False):
        _tracer.start_tracing()
        try:
            t1 = Tensor("dummy1", TensorConfig((3,), DType.Float32, device))
            t2 = Tensor("dummy2", TensorConfig((3,), DType.Float32, device))

            pbroadcast(t1, "x")
            pdot(t1, t2, "x")
            ppermute(t1, "x", [0, 1])
            pshuffle(t1, "x", [0, 1])
        finally:
            _tracer.stop_tracing()
