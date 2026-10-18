"""Ragged tensor representation with flat values and row split offsets."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Union

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor


class RaggedTensor:
    """Nested variable-length tensor represented by `values` and `row_splits`."""

    def __init__(self, values: Tensor, row_splits: Tensor) -> None:
        """Initialize `RaggedTensor`.

        Args:
            values (Tensor): Concatenated flat values tensor.
            row_splits (Tensor): 1D integer tensor of row boundary offsets.
        """
        self.values = values
        self.row_splits = row_splits

    @property
    def shape(self) -> Sequence[Union[int, str]]:
        """Return the ragged tensor shape with `-1` marking the ragged dimension.

        Returns:
            Sequence[Union[int, str]]: Shape tuple `(num_rows, -1, *inner_dims)`.
        """
        batch_size = max(0, len(self.row_splits) - 1)
        val_shape = list(self.values.shape)
        if val_shape:
            val_shape[0] = -1
        return tuple([batch_size] + val_shape)

    @property
    def dtype(self) -> DType:
        """Return the element data type of the ragged tensor.

        Returns:
            DType: Data type of `self.values`.
        """
        return self.values.dtype

    @property
    def device(self) -> Device:
        """Return the execution device of the ragged tensor.

        Returns:
            Device: Device of `self.values`.
        """
        return self.values.device

    @property
    def requires_grad(self) -> bool:
        """Return whether gradients should be tracked for `self.values`.

        Returns:
            bool: True if `self.values` requires gradients.
        """
        return self.values.requires_grad
