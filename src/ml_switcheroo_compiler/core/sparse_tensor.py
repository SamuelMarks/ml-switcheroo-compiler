"""Sparse tensor representations."""

from ml_switcheroo_compiler.core.tensor import Tensor


class SparseTensor:
    """Base class for sparse tensors."""

    def __init__(self, values: Tensor, dense_shape: tuple[int, ...]) -> None:
        """Init base."""
        self.values = values
        self.dense_shape = dense_shape
        self.format = "base"


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
