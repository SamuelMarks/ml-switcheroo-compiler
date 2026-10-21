"""Test module."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.core.ragged_tensor import RaggedTensor
from tests.testing_utils import as_callable, as_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


class DummyTensor:
    """Test class DummyTensor."""

    def __init__(self, shape: EagerValue, dtype: EagerValue = "float32", device: EagerValue = "cpu", requires_grad: EagerValue = True) -> None:
        """Test function __init__.

        Args:
            shape: Test parameter `shape`.
            dtype: Test parameter `dtype`.
            device: Test parameter `device`.
            requires_grad: Test parameter `requires_grad`.
        """
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.requires_grad = requires_grad

    def __len__(self) -> EagerValue:
        """Test function __len__."""
        return as_tensor(getattr(self.shape, "__getitem__")(0))


def test_ragged_tensor() -> None:
    """Test function test_ragged_tensor."""
    v = DummyTensor((4, 2))
    r = DummyTensor((3,))

    rt = as_callable(RaggedTensor)(v, r)
    assert rt.values == v
    assert rt.row_splits == r

    assert rt.shape == (2, -1, 2)
    assert rt.dtype == "float32"
    assert rt.device == "cpu"
    assert bool(rt.requires_grad) is True

    # Empty
    v2 = DummyTensor(())
    r2 = DummyTensor((0,))
    rt2 = as_callable(RaggedTensor)(v2, r2)
    assert rt2.shape == (0,)
