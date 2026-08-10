"""Tests for basic trigonometric operations in ops."""

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


def test_sin_op():
    a = Tensor(np.array([0, np.pi / 2, np.pi], dtype=np.float32), TensorConfig((3,), DType.Float32, Device("cpu")))
    res = dispatch_op("Sin", a)
    np.testing.assert_array_almost_equal(res.numpy(), np.array([0, 1, 0], dtype=np.float32), decimal=6)


def test_cos_op():
    a = Tensor(np.array([0, np.pi / 2, np.pi], dtype=np.float32), TensorConfig((3,), DType.Float32, Device("cpu")))
    res = dispatch_op("Cos", a)
    np.testing.assert_array_almost_equal(res.numpy(), np.array([1, 0, -1], dtype=np.float32), decimal=6)


def test_tan_op():
    a = Tensor(np.array([0, np.pi / 4], dtype=np.float32), TensorConfig((2,), DType.Float32, Device("cpu")))
    res = dispatch_op("Tan", a)
    np.testing.assert_array_almost_equal(res.numpy(), np.array([0, 1], dtype=np.float32), decimal=6)
