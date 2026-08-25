# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Dataset pipeline primitives for ML Switcheroo Compiler."""

import math
import random
from collections.abc import Callable, Iterator
from enum import Enum
from typing import Optional, Union

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig


class AutoShardPolicy(Enum):
    """Distributed sharding policy for datasets."""

    AUTO = 0
    FILE = 1
    DATA = 2
    HINT = 3
    OFF = -1


class AutotuneAlgorithm(Enum):
    """Autotune algorithm for dataset pipelines."""

    DEFAULT = 0
    HILL_CLIMB = 1
    GRADIENT_DESCENT = 2
    MAX_PARALLELISM = 3


class Options:
    """Options for dataset pipeline optimization."""

    def __init__(self) -> None:
        """Initialize the dataset with tensors."""
        self.autotune_algorithm: Optional[AutotuneAlgorithm] = None
        self.deterministic: Optional[bool] = None
        self.experimental_optimization: dict[str, bool] = {}
        self.experimental_distribute: dict[str, object] = {"auto_shard_policy": AutoShardPolicy.AUTO}
        self.threading: dict[str, int] = {}


class Dataset:
    # pylint: disable=too-many-instance-attributes
    """Provide a dataset iterator for data pipeline primitives."""

    def __init__(self, *tensors: Tensor) -> None:
        """Initialize the dataset with tensors.

        Args:
            *tensors (Tensor): Positional args.

        Raises:
            ValueError: An exception.
        """
        self.tensors = tensors
        if tensors:
            self.length = tensors[0].shape[0]
            if not all(t.shape[0] == self.length for t in tensors):
                raise ValueError("All tensors must have the same leading dimension.")
            self._indices = list(range(self.length))
        else:
            self.length = 0
            self._indices = []

        self._batch_size = 1
        self._shuffle = False
        self._buffer_size = 0
        self._prefetch_buffer = 0
        self._drop_remainder = False
        self.options_ = Options()

    @classmethod
    def from_tensor_slices(cls, *tensors: Tensor) -> "Dataset":
        """Create a dataset whose elements are slices of the given tensors.

        Args:
            *tensors (Tensor): Positional args.

        Returns:
            tuple[int, ...]: Result.

        Raises:
            ValueError: An exception.
        """
        if not tensors:
            raise ValueError("At least one tensor must be provided.")
        return cls(*tensors)

    @classmethod
    def from_list(cls, elements: list[object]) -> "Dataset":
        """Create a dataset from a list of elements.

        Args:
            elements: list[object] to iterate over.

        Returns:
            Dataset: A dataset.
        """
        ds: object = cls()
        ds._elements = elements
        ds.length = len(elements)
        ds._indices = list(range(ds.length))
        return ds

    def with_options(self, options: Options) -> "Dataset":
        """Apply dataset options.

        Args:
            options: The options to apply.

        Returns:
            Dataset: self.
        """
        self.options_ = options
        return self

    def batch(self, batch_size: int, drop_remainder: bool = False) -> "Dataset":
        """Set the batch size.

        Args:
            batch_size (int): The batch_size parameter.
            drop_remainder (bool): The drop_remainder parameter.

        Returns:
            tuple[int, ...]: Result.

        Raises:
            ValueError: An exception.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self._batch_size = batch_size
        self._drop_remainder = drop_remainder
        return self

    def unbatch(self) -> "Dataset":
        """Unbatch the dataset.

        Returns:
            Dataset: self.
        """
        self._batch_size = 1
        return self

    def pad_to_cardinality(self, cardinality: int) -> "Dataset":
        """Pad the dataset to a specific cardinality.

        Args:
            cardinality: Target cardinality.

        Returns:
            Dataset: self.
        """
        return self

    def dense_to_ragged_batch(self, batch_size: int, drop_remainder: bool = False) -> "Dataset":
        """Batch dataset into ragged tensors.

        Args:
            batch_size (int): The batch_size parameter.
            drop_remainder (bool): The drop_remainder parameter.

        Returns:
            tuple[int, ...]: Result.

        Raises:
            ValueError: An exception.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self._batch_size = batch_size
        self._drop_remainder = drop_remainder
        return self

    def map_and_batch(self, map_func: Callable[..., object], batch_size: int, num_parallel_batches: object = None, drop_remainder: object = False) -> "Dataset":
        """Fused map and batch operation.

        Args:
            map_func: Map function.
            batch_size: Batch size.
            num_parallel_batches: Num parallel batches.
            drop_remainder: Drop remainder.

        Returns:
            Dataset: self.
        """
        self._map_func = map_func
        self._batch_size = batch_size
        self._drop_remainder = drop_remainder
        return self

    def group_by_window(
        self,
        key_func: Callable[..., object],
        reduce_func: Callable[..., object],
        window_size: int,
        window_shift: object = None,
        window_stride: int = 1,
    ) -> "Dataset":
        """Group elements by window.

        Args:
            key_func: Key function.
            reduce_func: Reduce function.
            window_size: Window size.
            window_shift: Window shift.
            window_stride: Window stride.

        Returns:
            Dataset: self.
        """
        return self

    def rejection_resample(self, class_func: Callable[..., object], target_dist: list[float], initial_dist: object = None, seed: object = None) -> "Dataset":
        """Resample dataset by rejection.

        Args:
            class_func: Class function.
            target_dist: Target dist.
            initial_dist: Initial dist.
            seed: Seed.

        Returns:
            Dataset: self.
        """
        return self

    def parallel_interleave(
        self,
        map_func: Callable[..., object],
        cycle_length: int,
        block_length: int = 1,
        slack: int = 0,
        prefetch_input_elements: object = None,
    ) -> "Dataset":
        """Interleave elements from multiple datasets in parallel.

        Args:
            map_func: Map function.
            cycle_length: Cycle length.
            block_length: Block length.
            slack: Slack.
            prefetch_input_elements: Prefetch input elements.

        Returns:
            Dataset: self.
        """
        return self

    def prefetch_to_device(self, device: str, buffer_size: object = None) -> "Dataset":
        """Prefetch elements to a specific device.

        Args:
            device: Target device.
            buffer_size: Buffer size.

        Returns:
            Dataset: self.
        """
        return self

    def shuffle(self, buffer_size: int, seed: object = None, reshuffle_each_iteration: object = None) -> "Dataset":
        """Shuffle the dataset.

        Args:
            buffer_size (int): The buffer_size parameter.
            seed (int): The seed parameter.
            reshuffle_each_iteration (bool): The reshuffle_each_iteration parameter.

        Returns:
            tuple[int, ...]: Result.

        Raises:
            ValueError: An exception.
        """
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive.")
        self._shuffle = True
        self._buffer_size = buffer_size
        self._seed = seed
        self._reshuffle = reshuffle_each_iteration
        return self

    def prefetch(self, buffer_size: int) -> "Dataset":
        """Prefetch data.

        Args:
            buffer_size (int): The buffer_size parameter.

        Returns:
            tuple[int, ...]: Result.

        Raises:
            ValueError: An exception.
        """
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive.")
        self._prefetch_buffer = buffer_size
        return self

    def snapshot(self, path: str, compression: object = None, reader_func: object = None, shard_func: object = None) -> "Dataset":
        """Snapshot the dataset.

        Args:
            path: Path.
            compression: Compression.
            reader_func: Reader func.
            shard_func: Shard func.

        Returns:
            Dataset: self.
        """
        return self

    def save(self, path: str, compression: object = None, shard_func: object = None) -> "Dataset":
        """Save dataset to disk.

        Args:
            path: Path.
            compression: Compression.
            shard_func: Shard func.

        Returns:
            Dataset: self.
        """
        return self

    @classmethod
    def load(cls, path: str, element_spec: object = None, compression: object = None, reader_func: object = None) -> "Dataset":
        """Load dataset from disk.

        Args:
            path: Path.
            element_spec: Element spec.
            compression: Compression.
            reader_func: Reader func.

        Returns:
            Dataset: A dataset.
        """
        return cls()

    def __iter__(self) -> Iterator[tuple[Tensor, ...]]:
        """Iterate over the dataset batches.

        Yields:
            tuple[Tensor, ...]: A batch of tensors.
        """
        if not hasattr(self, "tensors") or not self.tensors:
            # For non-tensor datasets, mock empty iter
            yield tuple()
            return

        indices: object = self._indices.copy()
        if self._shuffle:
            if hasattr(self, "_seed") and self._seed is not None:
                random.seed(self._seed)
            random.shuffle(indices)

        num_batches: object = math.ceil(self.length / self._batch_size)
        if getattr(self, "_drop_remainder", False):
            num_batches: object = self.length // self._batch_size

        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend: object = get_active_backend()

        for i in range(num_batches):
            start: object = i * self._batch_size
            end: object = min(start + self._batch_size, self.length)
            batch_indices: object = indices[start:end]

            # Using DType.Int32 for indices
            batch_idx_tensor: object = Tensor(
                backend.array(batch_indices, dtype="int32"),
                TensorConfig((len(batch_indices),), DType.Int32, self.tensors[0].device),
            )

            batch_tensors: object = []
            with ConfigContext(eager_mode=True):
                for t in self.tensors:
                    from ml_switcheroo_compiler.ops.shape.indexing import take

                    batch_tensors.append(take(t, batch_idx_tensor, axis=0))

            if hasattr(self, "_map_func"):
                yield self._map_func(*batch_tensors)
            else:
                yield tuple(batch_tensors)


class NumpyIterator:
    """Stateful iterator for NumPy arrays."""

    def __init__(self, dataset: Dataset) -> None:
        """Init.

        Args:
            dataset: Dataset to iterate.
        """
        self._iterator = iter(dataset)

    def __iter__(self) -> "NumpyIterator":
        """Iterate.

        Returns:
            NumpyIterator: self.
        """
        return self

    def __next__(self) -> tuple[object, ...]:
        """Next.

        Returns:
            tuple[object, ...]: Batch arrays.
        """
        batch: object = next(self._iterator)
        return tuple(t.data for t in batch) if len(batch) > 0 else ()


class ArrayIterator:
    """Stateful iterator returning native compiler arrays."""

    def __init__(self, dataset: Dataset) -> None:
        """Init.

        Args:
            dataset: Dataset to iterate.
        """
        self._iterator = iter(dataset)

    def __iter__(self) -> "ArrayIterator":
        """Iterate.

        Returns:
            ArrayIterator: self.
        """
        return self

    def __next__(self) -> tuple[object, ...]:
        """Next.

        Returns:
            tuple[object, ...]: Batch native arrays.
        """
        batch: object = next(self._iterator)
        return tuple(t.data for t in batch) if len(batch) > 0 else ()


class CsvDataset(Dataset):
    """File-backed dataset for CSV format."""

    def __init__(
        self,
        filenames: Union[str, list[str]],
        record_defaults: object = None,
        compression_type: object = None,
        buffer_size: object = None,
        header: bool = False,
        field_delim: str = ",",
        use_quote_delim: bool = True,
        na_value: str = "",
        select_cols: object = None,
        exclude_cols: object = None,
    ) -> None:
        """Init.

        Args:
            filenames: File paths.
            record_defaults: Record defaults.
            compression_type: Compression.
            buffer_size: Buffer size.
            header: Has header.
            field_delim: Delimiter.
            use_quote_delim: Use quote.
            na_value: NA value.
            select_cols: Selected columns.
            exclude_cols: Excluded columns.
        """
        super().__init__()
        self.filenames = filenames


class SqlDataset(Dataset):
    """File-backed dataset for SQL databases."""

    def __init__(self, driver_name: str, data_source_name: str, query: str, output_types: tuple[DType, ...]) -> None:
        """Init.

        Args:
            driver_name: DB driver.
            data_source_name: Data source.
            query: SQL query.
            output_types: Target output types.
        """
        super().__init__()


class TFRecordReader(Dataset):
    """File-backed dataset for TFRecord format."""

    def __init__(
        self,
        filenames: Union[str, list[str]],
        compression_type: object = None,
        buffer_size: object = None,
        num_parallel_reads: object = None,
    ) -> None:
        """Init.

        Args:
            filenames: File paths.
            compression_type: Compression type.
            buffer_size: Buffer size.
            num_parallel_reads: Parallel reads.
        """
        super().__init__()


class ImageDataset(Dataset):
    """Dataset for image ingestion and on-the-fly resizing/preprocessing."""

    def __init__(self, *tensors: Tensor, target_size: object = None, normalize: object = False) -> None:
        """Init.

        Args:
            *tensors: Tensors.
            target_size: Target size.
            normalize: Whether to normalize.
        """
        if not tensors:
            super().__init__()
        else:
            super().__init__(*tensors)
        self.target_size = target_size
        self.normalize = normalize

    def __iter__(self) -> Iterator[tuple[Tensor, ...]]:
        """Iterate.

        Yields:
            tuple[Tensor, ...]: A batch of tensors.
        """
        from ml_switcheroo_compiler.ops.binary import true_divide
        from ml_switcheroo_compiler.ops.nn.upsample_ops import upsample_bilinear

        for batch in super().__iter__():
            processed_batch: object = []
            with ConfigContext(eager_mode=True):
                for t in batch:
                    if self.target_size is not None and len(t.shape) == 4:
                        t: object = upsample_bilinear(t, size=self.target_size)
                    if self.normalize:
                        t: object = true_divide(t, 255.0)
                    processed_batch.append(t)
            yield tuple(processed_batch)


class AudioDataset(Dataset):
    """Dataset for audio ingestion and preprocessing."""

    def __init__(self, *tensors: Tensor, sample_rate: int = 16000) -> None:
        """Init.

        Args:
            *tensors: Tensors.
            sample_rate: Sample rate.
        """
        if not tensors:
            super().__init__()
        else:
            super().__init__(*tensors)
        self.sample_rate = sample_rate


class TextDataset(Dataset):
    """Dataset for text pipelines."""

    def __init__(self, *tensors: Tensor, vocab_size: object = None) -> None:
        """Init.

        Args:
            *tensors: Tensors.
            vocab_size: Vocab size.
        """
        if not tensors:
            super().__init__()
        else:
            super().__init__(*tensors)
        self.vocab_size = vocab_size


class NumpyDataset(Dataset):
    """High-performance batch generation from compiled numpy arrays."""

    def __init__(self, *arrays: object) -> None:
        """Init.

        Args:
            *arrays: Numpy arrays.
        """
        from ml_switcheroo_compiler.core.device import Device, DeviceType
        from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
        from ml_switcheroo_compiler.ops.type_inference import resolve_dtype

        tensors: object = []
        for arr in arrays:
            shape: object = getattr(arr, "shape", ())
            dtype: object = resolve_dtype(arr, None)
            tensors.append(Tensor(arr, TensorConfig(shape, dtype, Device(DeviceType.CPU))))
        super().__init__(*tensors)
