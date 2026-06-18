from ml_switcheroo_compiler.ops.control_flow import vmap, pmap
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.tracing.tracer import _tracer


def test_vmap_pmap_scalar_fallback():
    def f(x, y):
        return x * y

    dev = Device("cpu")
    t = Tensor([1, 2, 3], (3,), DType.Int32, dev)

    with ConfigContext(eager_mode=False):
        _tracer.start_tracing()
        v_f = vmap(f, in_axes=(0, None))
        import pytest

        with pytest.raises(AttributeError):
            v_f(t, 5)  # 5 is non-tensor

        p_f = pmap(f)
        with pytest.raises(AttributeError):
            p_f(t, 5)  # 5 is non-tensor
        _tracer.stop_tracing()
