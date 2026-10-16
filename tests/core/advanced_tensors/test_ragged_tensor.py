"""Test module."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ml_switcheroo_compiler.core.ragged_tensor import RaggedTensor

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


class DummyTensor:
    """Test class DummyTensor."""

    def __init__(self, shape: EagerValue, dtype: EagerValue = "float32", device: EagerValue = "cpu", requires_grad: EagerValue = True) -> None:
        """Test function __init__."""
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.requires_grad = requires_grad

    def __len__(self) -> EagerValue:
        """Test function __len__."""
        return cast("EagerTensorProtocol", getattr(self.shape, "__getitem__")(0))


def test_ragged_tensor() -> None:
    """Test function test_ragged_tensor."""
    v = DummyTensor((4, 2))
    r = DummyTensor((3,))

    rt = cast("Callable[..., EagerTensorProtocol]", RaggedTensor)(v, r)
    assert rt.values == v
    assert rt.row_splits == r

    assert rt.shape == (2, -1, 2)
    assert rt.dtype == "float32"
    assert rt.device == "cpu"
    assert bool(rt.requires_grad) is True

    # Empty
    v2 = DummyTensor(())
    r2 = DummyTensor((0,))
    rt2 = cast("Callable[..., EagerTensorProtocol]", RaggedTensor)(v2, r2)
    assert rt2.shape == (0,)
