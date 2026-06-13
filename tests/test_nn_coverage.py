"""Provides required module functionality."""

from ml_switcheroo_compiler.nn import one_hot
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.config import config
import numpy as np


def test_nn_coverage_brute() -> None:
    """Execute the requested function."""
    config.eager_mode = True

    t_empty = Tensor(data=np.array(1), shape=(), dtype=DType.Int32, device=Device("cpu"))
    one_hot(t_empty, 5)

    t = Tensor(data=np.array([1, 2]), shape=(2,), dtype=DType.Int32, device=Device("cpu"))
    one_hot(t, 5)

    config.eager_mode = False
