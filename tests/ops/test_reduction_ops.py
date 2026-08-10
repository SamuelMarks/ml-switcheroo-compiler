"""Tests for basic reduction operations in ops."""

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


def test_reduce_sum_op():
    a = Tensor(np.array([[1, 2], [3, 4]], dtype=np.float32), TensorConfig((2, 2), DType.Float32, Device("cpu")))
    res = dispatch_op("Sum", a, axis=0)
    np.testing.assert_array_equal(res.numpy(), np.array([4, 6], dtype=np.float32))


def test_reduce_mean_op():
    a = Tensor(np.array([[1, 2], [3, 4]], dtype=np.float32), TensorConfig((2, 2), DType.Float32, Device("cpu")))
    res = dispatch_op("Mean", a, axis=1)
    np.testing.assert_array_equal(res.numpy(), np.array([1.5, 3.5], dtype=np.float32))


def test_reduce_max_op():
    a = Tensor(np.array([[1, 2], [3, 4]], dtype=np.float32), TensorConfig((2, 2), DType.Float32, Device("cpu")))
    res = dispatch_op("Max", a, axis=None)
    np.testing.assert_array_equal(res.numpy(), np.array(4, dtype=np.float32))


def test_reduce_min_op():
    a = Tensor(np.array([[1, 2], [3, 4]], dtype=np.float32), TensorConfig((2, 2), DType.Float32, Device("cpu")))
    res = dispatch_op("Min", a, axis=0)
    np.testing.assert_array_equal(res.numpy(), np.array([1, 2], dtype=np.float32))
