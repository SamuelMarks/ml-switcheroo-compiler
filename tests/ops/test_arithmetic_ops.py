"""Tests for basic arithmetic operations in ops."""

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


def test_add_op():
    a = Tensor(np.array([1, 2, 3], dtype=np.float32), TensorConfig((3,), DType.Float32, Device("cpu")))
    b = Tensor(np.array([4, 5, 6], dtype=np.float32), TensorConfig((3,), DType.Float32, Device("cpu")))

    res = dispatch_op("Add", a, b)
    np.testing.assert_array_equal(res.numpy(), np.array([5, 7, 9], dtype=np.float32))


def test_subtract_op():
    a = Tensor(np.array([1, 2, 3], dtype=np.float32), TensorConfig((3,), DType.Float32, Device("cpu")))
    b = Tensor(np.array([4, 5, 6], dtype=np.float32), TensorConfig((3,), DType.Float32, Device("cpu")))

    res = dispatch_op("Subtract", a, b)
    np.testing.assert_array_equal(res.numpy(), np.array([-3, -3, -3], dtype=np.float32))


def test_multiply_op():
    a = Tensor(np.array([1, 2, 3], dtype=np.float32), TensorConfig((3,), DType.Float32, Device("cpu")))
    b = Tensor(np.array([4, 5, 6], dtype=np.float32), TensorConfig((3,), DType.Float32, Device("cpu")))

    res = dispatch_op("Multiply", a, b)
    np.testing.assert_array_equal(res.numpy(), np.array([4, 10, 18], dtype=np.float32))


def test_true_divide_op():
    a = Tensor(np.array([4, 10, 18], dtype=np.float32), TensorConfig((3,), DType.Float32, Device("cpu")))
    b = Tensor(np.array([2, 5, 6], dtype=np.float32), TensorConfig((3,), DType.Float32, Device("cpu")))

    res = dispatch_op("TrueDivide", a, b)
    np.testing.assert_array_equal(res.numpy(), np.array([2, 2, 3], dtype=np.float32))


def test_negative_op():
    a = Tensor(np.array([1, -2, 3], dtype=np.float32), TensorConfig((3,), DType.Float32, Device("cpu")))

    res = dispatch_op("Negative", a)
    np.testing.assert_array_equal(res.numpy(), np.array([-1, 2, -3], dtype=np.float32))


def test_abs_op():
    a = Tensor(np.array([1, -2, -3], dtype=np.float32), TensorConfig((3,), DType.Float32, Device("cpu")))

    res = dispatch_op("Abs", a)
    np.testing.assert_array_equal(res.numpy(), np.array([1, 2, 3], dtype=np.float32))
