"""Sparse tensor."""

from ml_switcheroo_compiler.core.tensor import Tensor


class SparseTensor:
    """Represents a sparse tensor."""

    def __init__(self, indices: Tensor, values: Tensor, dense_shape: tuple[int, ...]) -> None:
        """Init."""
        self.indices = indices
        self.values = values
        self.dense_shape = dense_shape
