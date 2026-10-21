"""Tests for tracing utility functions."""

from __future__ import annotations

import typing

import pytest

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.control_flow_utils import _get_tensor_ids
from tests.testing_utils import as_callable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


class MockData:
    """Test class MockData."""

    def __init__(self, id_val: EagerValue = "mock") -> None:
        """Test function __init__.

        Args:
            id_val: Test parameter `id_val`.
        """
        self.id = id_val


def test_get_tensor_ids() -> None:
    """Test function test_get_tensor_ids."""
    t1 = as_callable(Tensor)(as_callable(MockData)("t1"), as_callable(TensorConfig)((), DType.Float32, as_callable(Device)("cpu")))
    t2 = as_callable(Tensor)(as_callable(MockData)("t2"), as_callable(TensorConfig)((), DType.Float32, as_callable(Device)("cpu")))

    assert _get_tensor_ids(t1) == ["t1"]
    assert _get_tensor_ids([t1, t2]) == ["t1", "t2"]
    assert _get_tensor_ids((t1, t2)) == ["t1", "t2"]

    with pytest.raises(TypeError):
        _get_tensor_ids({"a": t1, "b": t2})

    with pytest.raises(TypeError):
        _get_tensor_ids(1)
