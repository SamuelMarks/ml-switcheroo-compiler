"""Dataset pipeline primitives for ML Switcheroo Compiler."""

import math
import random
from collections.abc import Iterator

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig


class Dataset:
    """A dataset iterator for data pipeline primitives."""

    def __init__(self, *tensors: Tensor) -> None:
        """Initialize the dataset with tensors.

        Args:
            *tensors (Tensor): Tensors to iterate over.
        """
        if not tensors:
            raise ValueError("At least one tensor must be provided.")
        self.tensors = tensors
        self.length = tensors[0].shape[0]
        if not all(t.shape[0] == self.length for t in tensors):
            raise ValueError("All tensors must have the same leading dimension.")
        self._indices = list(range(self.length))
        self._batch_size = 1
        self._shuffle = False
        self._buffer_size = 0
        self._prefetch_buffer = 0

    def batch(self, batch_size: int) -> "Dataset":
        """Set the batch size.

        Args:
            batch_size (int): The batch size.

        Returns:
            Dataset: self
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self._batch_size = batch_size
        return self

    def shuffle(self, buffer_size: int) -> "Dataset":
        """Shuffle the dataset.

        Args:
            buffer_size (int): The shuffle buffer size.

        Returns:
            Dataset: self
        """
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive.")
        self._shuffle = True
        self._buffer_size = buffer_size
        return self

    def prefetch(self, buffer_size: int) -> "Dataset":
        """Prefetch data.

        Args:
            buffer_size (int): The prefetch buffer size.

        Returns:
            Dataset: self
        """
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive.")
        self._prefetch_buffer = buffer_size
        return self

    def __iter__(self) -> Iterator[tuple[Tensor, ...]]:
        """Iterate over the dataset batches.

        Yields:
            tuple[Tensor, ...]: A batch of tensors.
        """
        indices = self._indices.copy()
        if self._shuffle:
            # Simple global shuffle instead of buffer for now
            random.shuffle(indices)

        num_batches = math.ceil(self.length / self._batch_size)

        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()

        for i in range(num_batches):
            start = i * self._batch_size
            end = min(start + self._batch_size, self.length)
            batch_indices = indices[start:end]

            # Using DType.Int32 for indices
            batch_idx_tensor = Tensor(
                backend.array(batch_indices, dtype="int32"),
                TensorConfig((len(batch_indices),), DType.Int32, self.tensors[0].device),
            )

            batch_tensors = []
            with ConfigContext(eager_mode=True):
                for t in self.tensors:
                    from ml_switcheroo_compiler.ops.shape.indexing import gather

                    batch_tensors.append(gather(t, 0, batch_idx_tensor))

            yield tuple(batch_tensors)
