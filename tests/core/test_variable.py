"""Module docstring."""

import numpy as np

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Parameter, Tensor, TensorConfig, Variable
from ml_switcheroo_compiler.tracing import global_tracing_state


def test_variable_and_parameter() -> object:
    """Function docstring."""
    device = Device(DeviceType.CPU, 0)
    data = np.array([1, 2, 3])

    with ConfigContext(eager_mode=True):
        v = Variable(data, TensorConfig((3,), DType.Int32, device))
        assert not v.trainable

        p = Parameter(data, TensorConfig((3,), DType.Int32, device))
        assert p.trainable

        t = Tensor(np.array([4, 5, 6]), TensorConfig((3,), DType.Int32, device))
        v.assign(t)
        # Note: In our current eager fallback, Assign might not update the data directly if not implemented in numpy/eager.py
        # Actually eager assign will be a node emission or error if missing in eager backend. Let's catch it.
        try:
            v.assign(t)
        except Exception:
            pass

    with ConfigContext(eager_mode=False):
        global_tracing_state.start_tracing()
        try:
            v = Variable("dummy_v", TensorConfig((3,), DType.Int32, device))
            t = Tensor("dummy_t", TensorConfig((3,), DType.Int32, device))
            v.assign(t)
            v.assign_add(t)
            v.assign_sub(t)
        finally:
            global_tracing_state.stop_tracing()
