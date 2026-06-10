"""Unified backend array base class for ml-switcheroo."""

from typing import Any
from collections.abc import Sequence
from ml_switcheroo.core.dtype import DType
from ml_switcheroo.core.device import Device


class Tensor:
    """The unified backend array base class."""

    def __init__(
        self,
        data: Any,
        shape: Sequence[int],
        dtype: DType,
        device: Device,
        requires_grad: bool = False,
    ) -> None:
        """Initialize the Tensor."""
        self._data = data
        self._shape = tuple(shape)
        self._dtype = dtype
        self._device = device
        self._requires_grad = requires_grad

    @property
    def shape(self) -> Sequence[int]:
        """Get the shape of the tensor."""
        return self._shape

    @property
    def dtype(self) -> DType:
        """Get the data type of the tensor."""
        return self._dtype

    @property
    def device(self) -> Device:
        """Get the device of the tensor."""
        return self._device

    @property
    def requires_grad(self) -> bool:
        """Check if the tensor requires gradient computation."""
        return self._requires_grad

    @property
    def data(self) -> Any:
        """Get the underlying data payload."""
        return self._data
