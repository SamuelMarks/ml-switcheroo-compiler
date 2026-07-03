"""Module docstring."""

import numpy as np

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.control_flow import vmap


def test_vmap_in_axes_length_coverage() -> object:
    """Function docstring."""
    device = Device(DeviceType.CPU, 0)
    with ConfigContext(eager_mode=False):

        def my_func(a: object, b: object) -> object:
            """Function docstring."""
            return a + b

        t1 = Tensor(np.zeros((2, 3)), TensorConfig((2, 3), DType.Float32, device))
        t2 = Tensor(np.zeros((2, 3)), TensorConfig((2, 3), DType.Float32, device))

        # in_axes is a list/tuple but shorter than the number of arguments
        v_func = vmap(my_func, in_axes=(0,))

        # Tracing will happen
        try:
            v_func(t1, t2)
        except Exception:
            pass
