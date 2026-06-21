"""Provides required module functionality."""

import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.nn import one_hot


def test_one_hot() -> None:
    """Test one_hot execution and results."""
    config.eager_mode = True

    t_empty = Tensor(np.array(1), TensorConfig((), DType.Int32, Device("cpu")))
    res_empty = one_hot(t_empty, 5)
    assert res_empty.shape == (5,)
    assert res_empty.data is not None
    assert np.array_equal(res_empty.data, [0, 1, 0, 0, 0])

    t = Tensor(np.array([1, 2]), TensorConfig((2,), DType.Int32, Device("cpu")))
    res = one_hot(t, 5)
    assert res.shape == (2, 5)
    assert res.data is not None
    assert np.array_equal(res.data, [[0, 1, 0, 0, 0], [0, 0, 1, 0, 0]])

    config.eager_mode = False
