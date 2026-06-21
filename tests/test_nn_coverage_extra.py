"""Module docstring."""

import numpy as np

from ml_switcheroo_compiler.core import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.nn import gelu


def test_gelu_approximate() -> None:
    """Docstring."""
    config.eager_mode = True
    x = Tensor(np.array([-1.0, 0.0, 1.0]), TensorConfig((3,), DType.Float32, Device("cpu")))
    res = gelu(x, approximate=True)
    assert res is not None
