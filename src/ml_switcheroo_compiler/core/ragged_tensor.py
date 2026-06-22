"""Ragged tensor."""

from ml_switcheroo_compiler.core.tensor import Tensor


class RaggedTensor:
    """Represents a ragged tensor."""

    def __init__(self, values: Tensor, row_splits: Tensor) -> None:
        """Init."""
        self.values = values
        self.row_splits = row_splits
