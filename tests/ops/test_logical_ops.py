"""Tests for basic logical operations in ops."""

from __future__ import annotations

import typing

import numpy as np
import pytest

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

if typing.TYPE_CHECKING:
    from collections.abc import Iterator

    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol


@pytest.fixture(autouse=True)
def enable_eager_mode() -> Iterator[None]:
    """Test function enable_eager_mode."""
    config.eager_mode = True
    yield
    config.eager_mode = False


def test_equal_op() -> None:
    """Test function test_equal_op."""
    a = Tensor(np.array([1, 2, 3], dtype=np.float32), TensorConfig((3,), DType.Float32, Device("cpu")))
    b = Tensor(np.array([1, 5, 3], dtype=np.float32), TensorConfig((3,), DType.Float32, Device("cpu")))
    res = dispatch_op("Equal", a, b)
    typing.cast("EagerCallableProtocol", np.testing.assert_array_equal)(typing.cast("EagerCallableProtocol", typing.cast("EagerTensorProtocol", getattr(res, "numpy")))(), typing.cast("EagerCallableProtocol", np.array)([True, False, True], dtype=bool))


def test_greater_op() -> None:
    """Test function test_greater_op."""
    a = Tensor(np.array([4, 2, 3], dtype=np.float32), TensorConfig((3,), DType.Float32, Device("cpu")))
    b = Tensor(np.array([1, 5, 3], dtype=np.float32), TensorConfig((3,), DType.Float32, Device("cpu")))
    res = dispatch_op("Greater", a, b)
    typing.cast("EagerCallableProtocol", np.testing.assert_array_equal)(typing.cast("EagerCallableProtocol", typing.cast("EagerTensorProtocol", getattr(res, "numpy")))(), typing.cast("EagerCallableProtocol", np.array)([True, False, False], dtype=bool))


def test_logical_and_op() -> None:
    """Test function test_logical_and_op."""
    a = Tensor(np.array([True, True, False], dtype=bool), TensorConfig((3,), DType.Bool, Device("cpu")))
    b = Tensor(np.array([True, False, False], dtype=bool), TensorConfig((3,), DType.Bool, Device("cpu")))
    res = dispatch_op("LogicalAnd", a, b)
    typing.cast("EagerCallableProtocol", np.testing.assert_array_equal)(typing.cast("EagerCallableProtocol", typing.cast("EagerTensorProtocol", getattr(res, "numpy")))(), typing.cast("EagerCallableProtocol", np.array)([True, False, False], dtype=bool))


def test_logical_not_op() -> None:
    """Test function test_logical_not_op."""
    a = Tensor(np.array([True, False, False], dtype=bool), TensorConfig((3,), DType.Bool, Device("cpu")))
    res = dispatch_op("LogicalNot", a)
    typing.cast("EagerCallableProtocol", np.testing.assert_array_equal)(typing.cast("EagerCallableProtocol", typing.cast("EagerTensorProtocol", getattr(res, "numpy")))(), typing.cast("EagerCallableProtocol", np.array)([False, True, True], dtype=bool))
