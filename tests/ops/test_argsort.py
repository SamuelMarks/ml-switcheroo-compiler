import numpy as np

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.shape import ArgSort
from ml_switcheroo_compiler.ops.shape.frontend import argsort
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, _tracer


def test_argsort_opdef():
    s = ArgSort()
    assert s.infer_shape(None) == ()

    class DummyShape:
        shape = (10,)

    assert s.infer_shape(DummyShape()) == (10,)


def test_argsort_frontend():
    device = Device(DeviceType.CPU)
    x = Tensor(np.array([3, 1, 2]), TensorConfig((3,), DType.Int32, device))
    with ConfigContext(eager_mode=True):
        res = argsort(x)
        assert np.array_equal(res.data, [1, 2, 0])

    with ConfigContext(eager_mode=False):
        _tracer.start_tracing("test")
        x_proxy = Tensor(ProxyTensor("x", (3,), "int32"), TensorConfig((3,), DType.Int32, device))
        res_proxy = argsort(x_proxy)
        assert res_proxy.shape == (3,)
        _tracer.stop_tracing()
