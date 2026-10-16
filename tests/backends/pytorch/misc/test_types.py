"""Tests for PyTorch types module."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import pytest

import ml_switcheroo_compiler.backends.pytorch.types as pytorch_types
from ml_switcheroo_compiler.backends.pytorch.types import (
    array,
    asarray,
    item,
    zeros,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue

mock_torch = MagicMock()
mock_torch.zeros = lambda shape: f"zeros_{shape}"
mock_torch.tensor = lambda data, dtype=None: f"tensor_{data}_{dtype}"


class MockTensor:
    """Mock tensor."""

    def __init__(self, data: EagerValue) -> None:
        """Init."""
        self.data = data

    def item(self) -> EagerValue:
        """Item."""
        return cast("EagerTensorProtocol", getattr(self.data, "__getitem__")(0))


mock_torch.as_tensor = lambda data: MockTensor(data)
mock_torch.float32 = "mock_float32"
mock_torch.int32 = "mock_int32"
mock_torch.unknown_dtype = None


@pytest.fixture(autouse=True)
def _patch_pytorch_types() -> Iterator[None]:
    """Temporarily patch pytorch_types.torch and restore afterwards."""
    orig = cast("EagerTensorProtocol", getattr(pytorch_types, "torch"))
    setattr(pytorch_types, "torch", mock_torch)
    try:
        yield
    finally:
        setattr(pytorch_types, "torch", orig)


def test_zeros() -> None:
    """Test coverage."""
    res = cast("Callable[..., EagerTensorProtocol]", zeros)(None, (2, 2))
    assert res == "zeros_(2, 2)"


def test_array() -> None:
    """Test coverage."""
    res1 = cast("Callable[..., EagerTensorProtocol]", array)(None, [1, 2])
    assert res1 == "tensor_[1, 2]_None"

    res2 = cast("Callable[..., EagerTensorProtocol]", array)(None, [1, 2], dtype="float32")
    assert res2 == "tensor_[1, 2]_mock_float32"

    class MockDtype:
        """Test class MockDtype."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.value = "int32"

    res3 = cast("Callable[..., EagerTensorProtocol]", array)(None, [1, 2], dtype=cast("Callable[..., EagerTensorProtocol]", MockDtype)())
    assert res3 == "tensor_[1, 2]_mock_int32"

    res4 = cast("Callable[..., EagerTensorProtocol]", array)(None, [1, 2], dtype="unknown_dtype")
    assert res4 == "tensor_[1, 2]_None"


def test_asarray() -> None:
    """Test coverage."""
    res = cast("Callable[..., EagerTensorProtocol]", asarray)(None, [1, 2])
    assert cast("EagerTensorProtocol", getattr(res, "item"))() == 1


def test_item() -> None:
    """Test coverage."""
    res = cast("Callable[..., EagerTensorProtocol]", item)(None, [5])
    assert res == 5
