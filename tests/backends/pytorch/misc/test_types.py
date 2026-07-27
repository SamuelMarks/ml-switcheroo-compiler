"""Tests for PyTorch types module."""

from unittest.mock import MagicMock

mock_torch = MagicMock()
mock_torch.zeros = lambda shape: f"zeros_{shape}"
mock_torch.tensor = lambda data, dtype=None: f"tensor_{data}_{dtype}"


class MockTensor:
    """Mock tensor."""

    def __init__(self, data):
        """Init."""
        self.data = data

    def item(self):
        """Item."""
        return self.data[0]


mock_torch.as_tensor = lambda data: MockTensor(data)
mock_torch.float32 = "mock_float32"
mock_torch.int32 = "mock_int32"
mock_torch.unknown_dtype = None

import ml_switcheroo_compiler.backends.pytorch.types as pytorch_types

pytorch_types.torch = mock_torch

from ml_switcheroo_compiler.backends.pytorch.types import (
    array,
    asarray,
    item,
    zeros,
)


def test_zeros() -> None:
    """Test coverage."""
    res = zeros(None, (2, 2))
    assert res == "zeros_(2, 2)"


def test_array() -> None:
    """Test coverage."""
    res1 = array(None, [1, 2])
    assert res1 == "tensor_[1, 2]_None"

    res2 = array(None, [1, 2], dtype="float32")
    assert res2 == "tensor_[1, 2]_mock_float32"

    class MockDtype:
        def __init__(self):
            self.value = "int32"

    res3 = array(None, [1, 2], dtype=MockDtype())
    assert res3 == "tensor_[1, 2]_mock_int32"

    res4 = array(None, [1, 2], dtype="unknown_dtype")
    assert res4 == "tensor_[1, 2]_None"


def test_asarray() -> None:
    """Test coverage."""
    res = asarray(None, [1, 2])
    assert res.item() == 1


def test_item() -> None:
    """Test coverage."""
    res = item(None, [5])
    assert res == 5
