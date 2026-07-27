"""Sparse tensor representations."""

from collections.abc import Sequence
from typing import Union

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor


class SparseTensor:
    """Base class for sparse tensors."""

    def __init__(self, values: Tensor, dense_shape: tuple[int, ...]) -> None:
        """Init base."""
        self.values = values
        self.dense_shape = dense_shape
        self.format = "base"

    @property
    def shape(self) -> Sequence[Union[int, str]]:
        """Get the shape of the tensor."""
        return self.dense_shape

    @property
    def dtype(self) -> DType:
        """Get the data type of the tensor."""
        return self.values.dtype

    @property
    def device(self) -> Device:
        """Get the device of the tensor."""
        return self.values.device

    @property
    def requires_grad(self) -> bool:
        """Check if the tensor requires gradient computation."""
        return self.values.requires_grad


class SparseTensorCOO(SparseTensor):
    """Represents a sparse tensor in COO format."""

    def __init__(self, indices: Tensor, values: Tensor, dense_shape: tuple[int, ...]) -> None:
        """Init COO."""
        super().__init__(values, dense_shape)
        self.indices = indices
        self.format = "coo"


class SparseTensorCSR(SparseTensor):
    """Represents a sparse tensor in CSR format."""

    def __init__(
        self,
        row_pointers: Tensor,
        column_indices: Tensor,
        values: Tensor,
        dense_shape: tuple[int, ...],
    ) -> None:
        """Init CSR."""
        super().__init__(values, dense_shape)
        self.row_pointers = row_pointers
        self.column_indices = column_indices
        self.format = "csr"
