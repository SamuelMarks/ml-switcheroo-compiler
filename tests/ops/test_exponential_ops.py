"""Tests for basic exponential operations in ops."""

import numpy as np
import pytest

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.dispatcher import dispatch_op


@pytest.fixture(autouse=True)
def enable_eager_mode():
    config.eager_mode = True
    yield
    config.eager_mode = False


def test_exp_op():
    a = Tensor(np.array([0, 1, 2], dtype=np.float32), TensorConfig((3,), DType.Float32, Device("cpu")))
    res = dispatch_op("Exp", a)
    np.testing.assert_array_almost_equal(res.numpy(), np.array([1, np.e, np.e**2], dtype=np.float32), decimal=6)


def test_log_op():
    a = Tensor(np.array([1, np.e, np.e**2], dtype=np.float32), TensorConfig((3,), DType.Float32, Device("cpu")))
    res = dispatch_op("Log", a)
    np.testing.assert_array_almost_equal(res.numpy(), np.array([0, 1, 2], dtype=np.float32), decimal=6)


def test_pow_op():
    a = Tensor(np.array([1, 2, 3], dtype=np.float32), TensorConfig((3,), DType.Float32, Device("cpu")))
    b = Tensor(np.array([2, 3, 2], dtype=np.float32), TensorConfig((3,), DType.Float32, Device("cpu")))
    res = dispatch_op("Power", a, b)
    np.testing.assert_array_almost_equal(res.numpy(), np.array([1, 8, 9], dtype=np.float32), decimal=6)
