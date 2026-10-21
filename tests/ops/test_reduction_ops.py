"""Tests for basic reduction operations in ops."""

from __future__ import annotations

import typing

import numpy as np
import pytest

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.dispatcher import dispatch_op
from tests.testing_utils import as_callable, as_tensor

if typing.TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def enable_eager_mode() -> Iterator[None]:
    """Test function enable_eager_mode."""
    config.eager_mode = True
    yield
    config.eager_mode = False


def test_reduce_sum_op() -> None:
    """Test function test_reduce_sum_op."""
    a = Tensor(np.array([[1, 2], [3, 4]], dtype=np.float32), TensorConfig((2, 2), DType.Float32, Device("cpu")))
    res = dispatch_op("Sum", a, axis=0)
    as_callable(np.testing.assert_array_equal)(as_callable(as_tensor(getattr(res, "numpy")))(), as_callable(np.array)([4, 6], dtype=np.float32))


def test_reduce_mean_op() -> None:
    """Test function test_reduce_mean_op."""
    a = Tensor(np.array([[1, 2], [3, 4]], dtype=np.float32), TensorConfig((2, 2), DType.Float32, Device("cpu")))
    res = dispatch_op("Mean", a, axis=1)
    as_callable(np.testing.assert_array_equal)(as_callable(as_tensor(getattr(res, "numpy")))(), as_callable(np.array)([1.5, 3.5], dtype=np.float32))


def test_reduce_max_op() -> None:
    """Test function test_reduce_max_op."""
    a = Tensor(np.array([[1, 2], [3, 4]], dtype=np.float32), TensorConfig((2, 2), DType.Float32, Device("cpu")))
    res = dispatch_op("Max", a, axis=None)
    as_callable(np.testing.assert_array_equal)(as_callable(as_tensor(getattr(res, "numpy")))(), as_callable(np.array)(4, dtype=np.float32))


def test_reduce_min_op() -> None:
    """Test function test_reduce_min_op."""
    a = Tensor(np.array([[1, 2], [3, 4]], dtype=np.float32), TensorConfig((2, 2), DType.Float32, Device("cpu")))
    res = dispatch_op("Min", a, axis=0)
    as_callable(np.testing.assert_array_equal)(as_callable(as_tensor(getattr(res, "numpy")))(), as_callable(np.array)([1, 2], dtype=np.float32))
