"""Tests for basic arithmetic operations in ops."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


@pytest.fixture(autouse=True)
def enable_eager_mode() -> Iterator[None]:
    """Test function enable_eager_mode."""
    config.eager_mode = True
    yield
    config.eager_mode = False


def test_add_op() -> None:
    """Test function test_add_op."""
    a = Tensor(np.array([1, 2, 3], dtype=np.float32), TensorConfig((3,), DType.Float32, Device("cpu")))
    b = Tensor(np.array([4, 5, 6], dtype=np.float32), TensorConfig((3,), DType.Float32, Device("cpu")))

    res = dispatch_op("Add", a, b)
    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_array_equal)(cast("Callable[..., EagerTensorProtocol]", cast("EagerTensorProtocol", getattr(res, "numpy")))(), cast("Callable[..., EagerTensorProtocol]", np.array)([5, 7, 9], dtype=np.float32))


def test_subtract_op() -> None:
    """Test function test_subtract_op."""
    a = Tensor(np.array([1, 2, 3], dtype=np.float32), TensorConfig((3,), DType.Float32, Device("cpu")))
    b = Tensor(np.array([4, 5, 6], dtype=np.float32), TensorConfig((3,), DType.Float32, Device("cpu")))

    res = dispatch_op("Subtract", a, b)
    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_array_equal)(cast("Callable[..., EagerTensorProtocol]", cast("EagerTensorProtocol", getattr(res, "numpy")))(), cast("Callable[..., EagerTensorProtocol]", np.array)([-3, -3, -3], dtype=np.float32))


def test_multiply_op() -> None:
    """Test function test_multiply_op."""
    a = Tensor(np.array([1, 2, 3], dtype=np.float32), TensorConfig((3,), DType.Float32, Device("cpu")))
    b = Tensor(np.array([4, 5, 6], dtype=np.float32), TensorConfig((3,), DType.Float32, Device("cpu")))

    res = dispatch_op("Multiply", a, b)
    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_array_equal)(cast("Callable[..., EagerTensorProtocol]", cast("EagerTensorProtocol", getattr(res, "numpy")))(), cast("Callable[..., EagerTensorProtocol]", np.array)([4, 10, 18], dtype=np.float32))


def test_true_divide_op() -> None:
    """Test function test_true_divide_op."""
    a = Tensor(np.array([4, 10, 18], dtype=np.float32), TensorConfig((3,), DType.Float32, Device("cpu")))
    b = Tensor(np.array([2, 5, 6], dtype=np.float32), TensorConfig((3,), DType.Float32, Device("cpu")))

    res = dispatch_op("TrueDivide", a, b)
    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_array_equal)(cast("Callable[..., EagerTensorProtocol]", cast("EagerTensorProtocol", getattr(res, "numpy")))(), cast("Callable[..., EagerTensorProtocol]", np.array)([2, 2, 3], dtype=np.float32))


def test_negative_op() -> None:
    """Test function test_negative_op."""
    a = Tensor(np.array([1, -2, 3], dtype=np.float32), TensorConfig((3,), DType.Float32, Device("cpu")))

    res = dispatch_op("Negative", a)
    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_array_equal)(cast("Callable[..., EagerTensorProtocol]", cast("EagerTensorProtocol", getattr(res, "numpy")))(), cast("Callable[..., EagerTensorProtocol]", np.array)([-1, 2, -3], dtype=np.float32))


def test_abs_op() -> None:
    """Test function test_abs_op."""
    a = Tensor(np.array([1, -2, -3], dtype=np.float32), TensorConfig((3,), DType.Float32, Device("cpu")))

    res = dispatch_op("Abs", a)
    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_array_equal)(cast("Callable[..., EagerTensorProtocol]", cast("EagerTensorProtocol", getattr(res, "numpy")))(), cast("Callable[..., EagerTensorProtocol]", np.array)([1, 2, 3], dtype=np.float32))
