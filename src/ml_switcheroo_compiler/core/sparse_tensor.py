"""Coordinate (COO) and Compressed Sparse Row (CSR) sparse tensor representations."""

from __future__ import annotations

from collections.abc import Sequence

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor


class SparseTensor:
    """Base class for sparse tensor representations storing non-zero values and dense shape."""

    def __init__(self, values: Tensor, dense_shape: tuple[int, ...]) -> None:
        """Initialize `SparseTensor`.

        Args:
            values (Tensor): 1D tensor of non-zero element values.
            dense_shape (tuple[int, ...]): Full dense shape of the tensor.
        """
        self.values = values
        self.dense_shape = dense_shape
        self.format = "base"

    @property
    def shape(self) -> Sequence[int | str]:
        """Return the full dense shape of the sparse tensor.

        Returns:
            Sequence[Union[int, str]]: Dense shape tuple.
        """
        return self.dense_shape

    @property
    def dtype(self) -> DType:
        """Return the element data type of the sparse tensor.

        Returns:
            DType: Data type of `self.values`.
        """
        return self.values.dtype

    @property
    def device(self) -> Device:
        """Return the execution device where the sparse tensor resides.

        Returns:
            Device: Device of `self.values`.
        """
        return self.values.device

    @property
    def requires_grad(self) -> bool:
        """Return whether gradients should be computed for the sparse tensor's values.

        Returns:
            bool: True if `self.values` requires gradients.
        """
        return self.values.requires_grad


class SparseTensorCOO(SparseTensor):
    """Sparse tensor in Coordinate (COO) format."""

    def __init__(self, indices: Tensor, values: Tensor, dense_shape: tuple[int, ...]) -> None:
        """Initialize `SparseTensorCOO`.

        Args:
            indices (Tensor): 2D index tensor of shape `(nnz, ndim)`.
            values (Tensor): 1D tensor of non-zero values of length `nnz`.
            dense_shape (tuple[int, ...]): Full dense tensor shape.
        """
        super().__init__(values, dense_shape)
        self.indices = indices
        self.format = "coo"


class SparseTensorCSR(SparseTensor):
    """Sparse 2D matrix in Compressed Sparse Row (CSR) format."""

    def __init__(
        self,
        row_pointers: Tensor,
        column_indices: Tensor,
        values: Tensor,
        dense_shape: tuple[int, ...],
    ) -> None:
        """Initialize `SparseTensorCSR`.

        Args:
            row_pointers (Tensor): 1D row pointer offsets tensor of length `num_rows + 1`.
            column_indices (Tensor): 1D column index tensor of length `nnz`.
            values (Tensor): 1D non-zero values tensor of length `nnz`.
            dense_shape (tuple[int, ...]): 2D dense matrix shape `(num_rows, num_cols)`.
        """
        super().__init__(values, dense_shape)
        self.row_pointers = row_pointers
        self.column_indices = column_indices
        self.format = "csr"
