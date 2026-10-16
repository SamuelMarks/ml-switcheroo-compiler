"""Tests for test_type_inference."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

# ruff: noqa: E501
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.ops.type_inference import resolve_dtype, resolve_output_dtype_and_device

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


class MockTensor:
    """Test class MockTensor."""

    def __init__(self, dtype: EagerValue) -> None:
        """Test function __init__."""
        self.dtype = dtype


class MockData:
    """Test class MockData."""

    def __init__(self, dtype_str: EagerValue) -> None:
        """Test function __init__."""
        self.dtype = dtype_str


def test_resolve_dtype() -> None:
    """Test function test_resolve_dtype."""
    assert cast("Callable[..., EagerTensorProtocol]", resolve_dtype)(cast("Callable[..., EagerTensorProtocol]", MockData)("dtype('int32')"), None) == DType.Int32
    assert cast("Callable[..., EagerTensorProtocol]", resolve_dtype)(cast("Callable[..., EagerTensorProtocol]", MockData)("dtype_something"), None) == DType.Float32
    assert cast("Callable[..., EagerTensorProtocol]", resolve_dtype)(cast("Callable[..., EagerTensorProtocol]", MockData)("dtype=float32"), None) == DType.Float32
    assert cast("Callable[..., EagerTensorProtocol]", resolve_dtype)(cast("Callable[..., EagerTensorProtocol]", MockData)("invalid"), None) == DType.Float32
    assert cast("Callable[..., EagerTensorProtocol]", resolve_dtype)(None, cast("Callable[..., EagerTensorProtocol]", MockTensor)(DType.Int64)) == DType.Int64
    assert resolve_dtype(None, None) == DType.Float32


def test_resolve_output_dtype_and_device() -> None:
    """Test function test_resolve_output_dtype_and_device."""

    class MockTensorWithDevice:
        """Test class MockTensorWithDevice."""

        dtype = DType.Int32
        device = "cpu"

    t = MockTensorWithDevice()
    assert cast("Callable[..., EagerTensorProtocol]", resolve_output_dtype_and_device)(t, {"dtype": DType.Float64}) == (DType.Float64, "cpu")
    assert cast("Callable[..., EagerTensorProtocol]", resolve_output_dtype_and_device)(t, {}) == (DType.Int32, "cpu")
    assert resolve_output_dtype_and_device(None, {}) == (DType.Float32, None)
