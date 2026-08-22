# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Module sparse_tensor.py."""

from typing import Any

"""Sparse tensor representations."""

from collections.abc import Sequence
from typing import Union

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor


class SparseTensor:
    """Define base class for sparse tensors."""

    def __init__(self, values: Tensor, dense_shape: tuple[int, ...]) -> None:  # type: ignore
        """Init COO.

        Args:
            values (Tensor): The values parameter.
            dense_shape (tuple): The dense_shape parameter.
        """
        self.values = values
        self.dense_shape = dense_shape
        self.format = "base"

    @property
    def shape(self) -> Sequence[Union[int, str]]:
        """Get the shape of the tensor.

        Returns:
            tuple[int, ...]: Result.
        """
        return self.dense_shape

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


class SparseTensorCOO(SparseTensor):
    """Represents a sparse tensor in COO format."""

    def __init__(self, indices: Tensor, values: Tensor, dense_shape: tuple[int, ...]) -> None:  # type: ignore
        """Init COO.

        Args:
            indices (Tensor): The indices parameter.
            values (Tensor): The values parameter.
            dense_shape (tuple): The dense_shape parameter.
        """
        super().__init__(values, dense_shape)
        self.indices = indices
        self.format = "coo"


class SparseTensorCSR(SparseTensor):
    """Represents a sparse tensor in CSR format."""

    def __init__(
        self,
        row_pointers: Tensor,  # type: ignore
        column_indices: Tensor,  # type: ignore
        values: Tensor,  # type: ignore
        dense_shape: tuple[int, ...],
    ) -> None:
        """Init CSR.

        Args:
            row_pointers (Tensor): The row_pointers parameter.
            column_indices (Tensor): The column_indices parameter.
            values (Tensor): The values parameter.
            dense_shape (tuple): The dense_shape parameter.
        """
        super().__init__(values, dense_shape)
        self.row_pointers = row_pointers
        self.column_indices = column_indices
        self.format = "csr"
