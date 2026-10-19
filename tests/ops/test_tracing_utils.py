"""Tests for tracing utility functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.control_flow_utils import _get_tensor_ids

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerValue


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
    t1 = cast("EagerCallableProtocol", Tensor)(cast("EagerCallableProtocol", MockData)("t1"), cast("EagerCallableProtocol", TensorConfig)((), DType.Float32, cast("EagerCallableProtocol", Device)("cpu")))
    t2 = cast("EagerCallableProtocol", Tensor)(cast("EagerCallableProtocol", MockData)("t2"), cast("EagerCallableProtocol", TensorConfig)((), DType.Float32, cast("EagerCallableProtocol", Device)("cpu")))

    assert _get_tensor_ids(t1) == ["t1"]
    assert _get_tensor_ids([t1, t2]) == ["t1", "t2"]
    assert _get_tensor_ids((t1, t2)) == ["t1", "t2"]

    with pytest.raises(TypeError):
        _get_tensor_ids({"a": t1, "b": t2})

    with pytest.raises(TypeError):
        _get_tensor_ids(1)
