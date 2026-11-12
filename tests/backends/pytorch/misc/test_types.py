"""Tests for PyTorch types module."""

from __future__ import annotations

import typing
from unittest.mock import MagicMock

import pytest

import ml_switcheroo_compiler.backends.pytorch.types as pytorch_types
from ml_switcheroo_compiler.backends.pytorch.types import (
    array,
    asarray,
    item,
    zeros,
)
from tests.testing_utils import ensure_callable, ensure_eager_tensor, ensure_mutable

if typing.TYPE_CHECKING:
    from collections.abc import Iterator

    from ml_switcheroo_compiler.backends.eager_registry import EagerValue

mock_torch = MagicMock()
mock_torch.zeros = lambda shape: f"zeros_{shape}"
mock_torch.tensor = lambda data, dtype=None: f"tensor_{data}_{dtype}"


class MockTensor:
    """Mock tensor."""

    def __init__(self, data: EagerValue) -> None:
        """Init.

        Args:
            data: Test parameter `data`.
        """
        self.data = data

    def item(self) -> EagerValue:
        """Item.

        Returns:
            EagerValue: Computed output value from `item`.
        """
        return ensure_eager_tensor(self.data)[0]


mock_torch.as_tensor = lambda data: MockTensor(data)
mock_torch.float32 = "mock_float32"
mock_torch.int32 = "mock_int32"
mock_torch.unknown_dtype = None


@pytest.fixture(autouse=True)
def _patch_pytorch_types() -> Iterator[None]:
    """Temporarily patch pytorch_types.torch and restore afterwards.

    Yields:
        None: Yielded test context or value from `_patch_pytorch_types`.
    """
    orig = ensure_eager_tensor(ensure_eager_tensor(pytorch_types).torch)
    ensure_mutable(pytorch_types).torch = ensure_eager_tensor(mock_torch)
    try:
        yield
    finally:
        ensure_mutable(pytorch_types).torch = ensure_eager_tensor(orig)


def test_zeros() -> None:
    """Test coverage."""
    res = ensure_callable(zeros)(None, (2, 2))
    assert res == "zeros_(2, 2)"


def test_array() -> None:
    """Test coverage."""
    res1 = ensure_callable(array)(None, [1, 2])
    assert res1 == "tensor_[1, 2]_None"

    res2 = ensure_callable(array)(None, [1, 2], dtype="float32")
    assert res2 == "tensor_[1, 2]_mock_float32"

    class MockDtype:
        """Test class MockDtype."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.value = "int32"

    res3 = ensure_callable(array)(None, [1, 2], dtype=ensure_callable(MockDtype)())
    assert res3 == "tensor_[1, 2]_mock_int32"

    res4 = ensure_callable(array)(None, [1, 2], dtype="unknown_dtype")
    assert res4 == "tensor_[1, 2]_None"


def test_asarray() -> None:
    """Test coverage."""
    res = ensure_callable(asarray)(None, [1, 2])
    assert ensure_eager_tensor(ensure_eager_tensor(res).item)() == 1


def test_item() -> None:
    """Test coverage."""
    res = ensure_callable(item)(None, [5])
    assert res == 5
