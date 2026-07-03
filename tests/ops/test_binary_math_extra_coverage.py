"""Module docstring."""

import numpy as np

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.binary.math import Betainc


def test_betainc_infer_shape() -> object:
    """Function docstring."""
    device = Device("cpu")
    t1 = Tensor(np.ones((2,)), TensorConfig((2,), "float32", device))
    t2 = Tensor(np.ones((2,)), TensorConfig((2,), "float32", device))
    opdef = Betainc()
    # just trigger it
    opdef.infer_shape(t1, t2)
