"""Tests for test_type_inference."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

# ruff: noqa: E501
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.ops.type_inference import resolve_dtype, resolve_output_dtype_and_device

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerValue


class MockTensor:
    """Test class MockTensor."""

    def __init__(self, dtype: EagerValue) -> None:
        """Test function __init__.

        Args:
            dtype: Test parameter `dtype`.
        """
        self.dtype = dtype


class MockData:
    """Test class MockData."""

    def __init__(self, dtype_str: EagerValue) -> None:
        """Test function __init__.

        Args:
            dtype_str: Test parameter `dtype_str`.
        """
        self.dtype = dtype_str


def test_resolve_dtype() -> None:
    """Test function test_resolve_dtype."""
    assert cast("EagerCallableProtocol", resolve_dtype)(cast("EagerCallableProtocol", MockData)("dtype('int32')"), None) == DType.Int32
    assert cast("EagerCallableProtocol", resolve_dtype)(cast("EagerCallableProtocol", MockData)("dtype_something"), None) == DType.Float32
    assert cast("EagerCallableProtocol", resolve_dtype)(cast("EagerCallableProtocol", MockData)("dtype=float32"), None) == DType.Float32
    assert cast("EagerCallableProtocol", resolve_dtype)(cast("EagerCallableProtocol", MockData)("invalid"), None) == DType.Float32
    assert cast("EagerCallableProtocol", resolve_dtype)(None, cast("EagerCallableProtocol", MockTensor)(DType.Int64)) == DType.Int64
    assert resolve_dtype(None, None) == DType.Float32


def test_resolve_output_dtype_and_device() -> None:
    """Test function test_resolve_output_dtype_and_device."""

    class MockTensorWithDevice:
        """Test class MockTensorWithDevice."""

        dtype = DType.Int32
        device = "cpu"

    t = MockTensorWithDevice()
    assert cast("EagerCallableProtocol", resolve_output_dtype_and_device)(t, {"dtype": DType.Float64}) == (DType.Float64, "cpu")
    assert cast("EagerCallableProtocol", resolve_output_dtype_and_device)(t, {}) == (DType.Int32, "cpu")
    assert resolve_output_dtype_and_device(None, {}) == (DType.Float32, None)
