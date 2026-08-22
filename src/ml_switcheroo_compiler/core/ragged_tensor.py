# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Module ragged_tensor.py."""

from typing import Any

"""Ragged tensor."""

from collections.abc import Sequence
from typing import Union

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor


class RaggedTensor:
    """Represents a ragged tensor."""

    def __init__(self, values: Tensor, row_splits: Tensor) -> None:  # type: ignore
        """Init.

        Args:
            values (Tensor): The values parameter.
            row_splits (Tensor): The row_splits parameter.
        """
        self.values = values
        self.row_splits = row_splits

    @property
    def shape(self) -> Sequence[Union[int, str]]:
        """Get the shape of the tensor.

        Returns:
            tuple[int, ...]: Result.
        """
        # The true shape of a ragged tensor is typically (batch, None, ...)
        # We approximate it by taking the number of rows from row_splits
        # and marking the next dimension as 'None' (symbolic) if possible,
        # or just delegating to values. For uniformity, returning a tuple
        batch_size = max(0, len(self.row_splits) - 1)
        val_shape = list(self.values.shape)
        if val_shape:
            val_shape[0] = -1
        return tuple([batch_size] + val_shape)

    @property
    def dtype(self) -> DType:
        """Get the data type of the tensor.

        Returns:
        DType: Result.
        """
        return self.values.dtype

    @property
    def device(self) -> Device:
        """Get the device of the tensor.

        Returns:
        Device: Result.
        """
        return self.values.device

    @property
    def requires_grad(self) -> bool:
        """Check if the tensor requires gradient computation.

        Returns:
        bool: Result.
        """
        return self.values.requires_grad
