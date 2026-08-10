"""Tests for basic matrix operations in ops."""

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


def test_matmul_op():
    a = Tensor(np.array([[1, 2], [3, 4]], dtype=np.float32), TensorConfig((2, 2), DType.Float32, Device("cpu")))
    b = Tensor(np.array([[5, 6], [7, 8]], dtype=np.float32), TensorConfig((2, 2), DType.Float32, Device("cpu")))
    res = dispatch_op("Dot", a, b)
    np.testing.assert_array_equal(res.numpy(), np.array([[19, 22], [43, 50]], dtype=np.float32))


def test_transpose_op():
    a = Tensor(np.array([[1, 2], [3, 4]], dtype=np.float32), TensorConfig((2, 2), DType.Float32, Device("cpu")))
    res = dispatch_op("Transpose", a)
    np.testing.assert_array_equal(res.numpy(), np.array([[1, 3], [2, 4]], dtype=np.float32))
