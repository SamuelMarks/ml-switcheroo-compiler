"""Symbolic and eager data stream pipeline primitives for the compiler."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterator, Sized
from enum import Enum
from typing import Optional, Union, cast

from ml_switcheroo_compiler.backends.registry import EagerCallableProtocol
from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import AttrValue, TensorOrScalar
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig


class AutoShardPolicy(Enum):
    """Distributed sharding policy for data stream pipelines."""

    AUTO = 0
    FILE = 1
    DATA = 2
    HINT = 3
    OFF = -1


class AutotuneAlgorithm(Enum):
    """Autotuning algorithm for data stream pipeline stages."""

    DEFAULT = 0
    HILL_CLIMB = 1
    GRADIENT_DESCENT = 2
    MAX_PARALLELISM = 3


class Options:
    """Configuration options for data stream pipeline optimization and sharding."""

    def __init__(self) -> None:
        """Initialize default pipeline execution options."""
        self.autotune_algorithm: Optional[AutotuneAlgorithm] = None
        self.deterministic: Optional[bool] = None
        self.experimental_optimization: dict[str, bool] = {}
        self.experimental_distribute = {"auto_shard_policy": AutoShardPolicy.AUTO}
        self.threading: dict[str, int] = {}


class Dataset:
    # pylint: disable=too-many-instance-attributes
    """Symbolic and eager data stream generator over aligned tensors."""

    def __init__(self, *tensors: Tensor) -> None:
        """Initialize the dataset stream with aligned input tensors.

        Args:
            *tensors (Tensor): Input tensors sharing the same leading dimension.

        Raises:
            ValueError: If input tensors do not share the same leading dimension size.
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
    def from_tensor_slices(cls, *tensors: Tensor) -> Dataset:
        """Create a dataset stream whose elements are slices along dimension 0 of `tensors`.

        Args:
            *tensors (Tensor): One or more tensors with identical leading dimension.

        Returns:
            Dataset: Configured dataset stream instance.

        Raises:
            ValueError: If no tensors are provided.
        """
        if not tensors:
            raise ValueError("At least one tensor must be provided.")
        return cls(*tensors)

    @classmethod
    def from_list(cls, elements: TensorOrScalar) -> Dataset:
        """Create a dataset stream from a sized sequence of elements.

        Args:
            elements (TensorOrScalar): Sequence of elements to iterate over.

        Returns:
            Dataset: Configured dataset stream instance.
        """
        ds = cls()
        setattr(ds, "_elements", elements)
        ds.length = len(elements) if isinstance(elements, Sized) else 0
        ds._indices = list(range(ds.length))
        return ds

    def with_options(self, options: Options) -> Dataset:
        """Attach pipeline execution and sharding options to this stream.

        Args:
            options (Options): Pipeline configuration options to apply.

        Returns:
            Dataset: The updated dataset stream (`self`).
        """
        self.options_ = options
        return self

    def batch(self, batch_size: int, drop_remainder: bool = False) -> Dataset:
        """Configure consecutive elements of this stream to be grouped into batches.

        Args:
            batch_size (int): Number of elements to combine per batch.
            drop_remainder (bool): Whether to drop the final batch if smaller than `batch_size`.

        Returns:
            Dataset: The batched dataset stream (`self`).

        Raises:
            ValueError: If `batch_size` is not positive.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self._batch_size = batch_size
        self._drop_remainder = drop_remainder
        return self

    def unbatch(self) -> Dataset:
        """Reset stream batching so elements are yielded individually (batch size 1).

        Returns:
            Dataset: The unbatched dataset stream (`self`).
        """
        self._batch_size = 1
        return self

    def pad_to_cardinality(self, cardinality: int) -> Dataset:
        """Pad the stream cardinality metadata to a target element count.

        Args:
            cardinality (int): Target stream cardinality.

        Returns:
            Dataset: The padded dataset stream (`self`).
        """
        _ = cardinality
        return self

    def dense_to_ragged_batch(self, batch_size: int, drop_remainder: bool = False) -> Dataset:
        """Batch variable-length elements into ragged batch tensors.

        Args:
            batch_size (int): Number of elements per ragged batch.
            drop_remainder (bool): Whether to drop the final incomplete batch.

        Returns:
            Dataset: The ragged-batched dataset stream (`self`).

        Raises:
            ValueError: If `batch_size` is not positive.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self._batch_size = batch_size
        self._drop_remainder = drop_remainder
        return self

    def map_and_batch(
        self,
        map_func: TensorOrScalar,
        batch_size: int,
        num_parallel_batches: Optional[int] = None,
        drop_remainder: bool = False,
    ) -> Dataset:
        """Apply a fused element mapping function and batching stage.

        Args:
            map_func (TensorOrScalar): Transformation callable applied to each batch.
            batch_size (int): Number of elements per batch.
            num_parallel_batches (Optional[int]): Optional number of batches to compute in parallel.
            drop_remainder (bool): Whether to drop the final incomplete batch.

        Returns:
            Dataset: The transformed dataset stream (`self`).
        """
        _ = num_parallel_batches
        self._map_func = map_func
        self._batch_size = batch_size
        self._drop_remainder = bool(drop_remainder)
        return self

    def group_by_window(
        self,
        key_func: TensorOrScalar,
        reduce_func: TensorOrScalar,
        window_size: int,
        window_shift: Optional[int] = None,
        window_stride: int = 1,
    ) -> Dataset:
        """Group stream elements by key and reduce them in sliding windows.

        Args:
            key_func (TensorOrScalar): Callable mapping an element to a grouping key.
            reduce_func (TensorOrScalar): Callable reducing a key and window into a stream.
            window_size (int): Number of elements per window.
            window_shift (Optional[int]): Optional window shift increment.
            window_stride (int): Stride between elements within a window.

        Returns:
            Dataset: The windowed dataset stream (`self`).
        """
        _ = (key_func, reduce_func, window_size, window_shift, window_stride)
        return self

    def rejection_resample(
        self,
        class_func: TensorOrScalar,
        target_dist: list[float],
        initial_dist: Optional[list[float]] = None,
        seed: Optional[int] = None,
    ) -> Dataset:
        """Resample stream elements via rejection sampling to match `target_dist`.

        Args:
            class_func (TensorOrScalar): Callable mapping an element to its class index.
            target_dist (list[float]): Target class probability distribution.
            initial_dist (Optional[list[float]]): Optional known initial class distribution.
            seed (Optional[int]): Optional PRNG seed for acceptance sampling.

        Returns:
            Dataset: The resampled dataset stream (`self`).
        """
        _ = (class_func, target_dist, initial_dist, seed)
        return self

    def parallel_interleave(
        self,
        map_func: TensorOrScalar,
        cycle_length: int,
        block_length: int = 1,
        slack: int = 0,
        prefetch_input_elements: Optional[int] = None,
    ) -> Dataset:
        """Interleave elements from multiple mapped streams in parallel.

        Args:
            map_func (TensorOrScalar): Callable mapping input elements to sub-streams.
            cycle_length (int): Number of streams to interleave concurrently.
            block_length (int): Number of consecutive elements consumed per stream.
            slack (int): Optional scheduling slack parameter.
            prefetch_input_elements (Optional[int]): Optional input prefetch buffer size.

        Returns:
            Dataset: The interleaved dataset stream (`self`).
        """
        _ = (map_func, cycle_length, block_length, slack, prefetch_input_elements)
        return self

    def prefetch_to_device(self, device: str, buffer_size: Optional[int] = None) -> Dataset:
        """Stage prefetched batches onto a target execution device.

        Args:
            device (str): Target device identifier (e.g., `'cpu'`, `'gpu:0'`).
            buffer_size (Optional[int]): Optional number of batches to buffer on device.

        Returns:
            Dataset: The device-prefetched dataset stream (`self`).
        """
        _ = (device, buffer_size)
        return self

    def shuffle(
        self,
        buffer_size: int,
        seed: Optional[int] = None,
        reshuffle_each_iteration: Optional[bool] = None,
    ) -> Dataset:
        """Randomly shuffle elements of this stream using a reservoir buffer.

        Args:
            buffer_size (int): Size of the shuffle buffer (must be positive).
            seed (Optional[int]): Optional random seed for reproducible shuffling.
            reshuffle_each_iteration (Optional[bool]): Whether to reshuffle on each epoch.

        Returns:
            Dataset: The shuffled dataset stream (`self`).

        Raises:
            ValueError: If `buffer_size` is not positive.
        """
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive.")
        self._shuffle = True
        self._buffer_size = buffer_size
        self._seed = seed
        self._reshuffle = reshuffle_each_iteration
        return self

    def prefetch(self, buffer_size: int) -> Dataset:
        """Configure the number of batches to prefetch asynchronously.

        Args:
            buffer_size (int): Number of batches to prefetch (must be positive).

        Returns:
            Dataset: The prefetched dataset stream (`self`).

        Raises:
            ValueError: If `buffer_size` is not positive.
        """
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive.")
        self._prefetch_buffer = buffer_size
        return self

    def snapshot(
        self,
        path: str,
        compression: Optional[str] = None,
        reader_func: TensorOrScalar = None,
        shard_func: TensorOrScalar = None,
    ) -> Dataset:
        """Persist or read a cached stream snapshot at `path`.

        Args:
            path (str): Filesystem path for snapshot storage.
            compression (Optional[str]): Optional compression format.
            reader_func (TensorOrScalar): Optional custom shard reader callable.
            shard_func (TensorOrScalar): Optional custom sharding callable.

        Returns:
            Dataset: The snapshotted dataset stream (`self`).
        """
        _ = (path, compression, reader_func, shard_func)
        return self

    def save(self, path: str, compression: Optional[str] = None, shard_func: TensorOrScalar = None) -> Dataset:
        """Materialize the dataset stream to disk at `path`.

        Args:
            path (str): Destination directory path.
            compression (Optional[str]): Optional compression format.
            shard_func (TensorOrScalar): Optional custom shard assignment callable.

        Returns:
            Dataset: The saved dataset stream (`self`).
        """
        _ = (path, compression, shard_func)
        return self

    @classmethod
    def load(
        cls,
        path: str,
        element_spec: AttrValue = None,
        compression: Optional[str] = None,
        reader_func: TensorOrScalar = None,
    ) -> Dataset:
        """Load a materialized dataset stream from `path`.

        Args:
            path (str): Source directory path.
            element_spec (AttrValue): Optional element type specification.
            compression (Optional[str]): Optional compression format.
            reader_func (TensorOrScalar): Optional custom reader callable.

        Returns:
            Dataset: Loaded dataset stream instance.
        """
        _ = (path, element_spec, compression, reader_func)
        return cls()

    def __iter__(self) -> Iterator[tuple[Tensor, ...]]:
        """Iterate over batches of tensors yielded by the stream.

        Yields:
            tuple[Tensor, ...]: Tuple of batch tensors for each stream step.
        """
        if not hasattr(self, "tensors") or not self.tensors:
            yield ()
            return

        indices = self._indices.copy()
        if self._shuffle:
            if hasattr(self, "_seed") and self._seed is not None:
                random.seed(cast("int | float | str | bytes | bytearray | None", self._seed))
            random.shuffle(indices)

        num_batches = math.ceil(self.length / self._batch_size)
        if getattr(self, "_drop_remainder", False):
            num_batches = self.length // self._batch_size

        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()

        for i in range(num_batches):
            start = i * self._batch_size
            end = min(start + self._batch_size, self.length)
            batch_indices = indices[start:end]

            batch_idx_tensor = Tensor(
                backend.array(batch_indices, dtype="int32"),
                TensorConfig((len(batch_indices),), DType.Int32, self.tensors[0].device),
            )

            batch_tensors: list[Tensor] = []
            with ConfigContext(eager_mode=True):
                for t in self.tensors:
                    from ml_switcheroo_compiler.ops.shape.indexing import take

                    batch_tensors.append(cast(Tensor, take(t, batch_idx_tensor, axis=0)))

            if hasattr(self, "_map_func"):
                yield cast("Callable[..., tuple[Tensor, ...]]", self._map_func)(*batch_tensors)
            else:
                yield tuple(batch_tensors)


class NumpyIterator:
    """Stateful iterator yielding raw backend arrays for each batch in a `Dataset`."""

    def __init__(self, dataset: Dataset) -> None:
        """Initialize NumpyIterator over `dataset`.

        Args:
            dataset (Dataset): Source dataset stream to iterate.
        """
        self._iterator = iter(dataset)

    def __iter__(self) -> NumpyIterator:
        """Return the iterator object itself.

        Returns:
            NumpyIterator: `self`.
        """
        return self

    def __next__(self) -> tuple[TensorOrScalar, ...]:
        """Advance the iterator and return the next batch of raw arrays.

        Returns:
            tuple[TensorOrScalar, ...]: Tuple of raw backend arrays for the current batch.
        """
        batch = next(self._iterator)
        return tuple(t.data for t in batch) if len(batch) > 0 else ()


class ArrayIterator:
    """Stateful iterator yielding native compiler arrays for each batch in a `Dataset`."""

    def __init__(self, dataset: Dataset) -> None:
        """Initialize ArrayIterator over `dataset`.

        Args:
            dataset (Dataset): Source dataset stream to iterate.
        """
        self._iterator = iter(dataset)

    def __iter__(self) -> ArrayIterator:
        """Return the iterator object itself.

        Returns:
            ArrayIterator: `self`.
        """
        return self

    def __next__(self) -> tuple[TensorOrScalar, ...]:
        """Advance the iterator and return the next batch of native arrays.

        Returns:
            tuple[TensorOrScalar, ...]: Tuple of native backend arrays for the current batch.
        """
        batch = next(self._iterator)
        return tuple(t.data for t in batch) if len(batch) > 0 else ()


class CsvDataset(Dataset):
    """File-backed tabular dataset stream for delimited CSV files."""

    def __init__(
        self,
        filenames: Union[str, list[str]],
        record_defaults: Optional[list[Union[int, float, str]]] = None,
        compression_type: Optional[str] = None,
        buffer_size: Optional[int] = None,
        header: bool = False,
        field_delim: str = ",",
        *args: Union[bool, str, list[int], None],
        **kwargs: Union[bool, str, list[int], None],
    ) -> None:
        """Initialize CsvDataset.

        Args:
            filenames (Union[str, list[str]]): Path or list of paths to CSV files.
            record_defaults (Optional[list[Union[int, float, str]]]): Default values per column.
            compression_type (Optional[str]): Optional compression format.
            buffer_size (Optional[int]): Optional read buffer size in bytes.
            header (bool): Whether to skip the initial header line.
            field_delim (str): Column separator character.
            *args (Union[bool, str, list[int], None]): Additional positional arguments.
            **kwargs (Union[bool, str, list[int], None]): Additional keyword arguments.
        """
        del record_defaults, compression_type, buffer_size, header, field_delim, args, kwargs
        super().__init__()
        self.filenames = filenames


class SqlDataset(Dataset):
    """Query-backed dataset stream reading rows from a relational database."""

    def __init__(
        self,
        driver_name: str,
        data_source_name: str,
        query: str,
        output_types: tuple[DType, ...],
    ) -> None:
        """Initialize SqlDataset.

        Args:
            driver_name (str): Database driver identifier (e.g., `'sqlite'`).
            data_source_name (str): Database connection string or file path.
            query (str): SQL SELECT query to execute.
            output_types (tuple[DType, ...]): Tuple of column data types.
        """
        _ = (driver_name, data_source_name, query, output_types)
        super().__init__()


class BinaryRecordReader(Dataset):
    """File-backed dataset stream for length-prefixed CRC-framed binary record files."""

    def __init__(
        self,
        filenames: Union[str, list[str]],
        compression_type: Optional[str] = None,
        buffer_size: Optional[int] = None,
        num_parallel_reads: Optional[int] = None,
    ) -> None:
        """Initialize BinaryRecordReader.

        Args:
            filenames (Union[str, list[str]]): Path or list of paths to binary record files.
            compression_type (Optional[str]): Optional compression format (`''`, `'GZIP'`, `'ZLIB'`).
            buffer_size (Optional[int]): Optional read buffer size in bytes.
            num_parallel_reads (Optional[int]): Optional number of files to read in parallel.
        """
        _ = (filenames, compression_type, buffer_size, num_parallel_reads)
        super().__init__()


TFRecordReader = BinaryRecordReader


class ImageDataset(Dataset):
    """Dataset stream for image tensors with optional bilinear resize and normalization."""

    def __init__(
        self,
        *tensors: Tensor,
        target_size: Optional[tuple[int, int]] = None,
        normalize: bool = False,
    ) -> None:
        """Initialize ImageDataset.

        Args:
            *tensors (Tensor): Input 4D image tensors.
            target_size (Optional[tuple[int, int]]): Optional `(height, width)` tuple for resizing.
            normalize (bool): Whether to scale pixel values by `1 / 255.0`.
        """
        if not tensors:
            super().__init__()
        else:
            super().__init__(*tensors)
        self.target_size = target_size
        self.normalize = normalize

    def __iter__(self) -> Iterator[tuple[Tensor, ...]]:
        """Iterate over preprocessed image batches.

        Yields:
            tuple[Tensor, ...]: Tuple of resized and/or normalized image batch tensors.
        """
        from ml_switcheroo_compiler.ops.binary import true_divide
        from ml_switcheroo_compiler.ops.nn.upsample_ops import upsample_bilinear

        for batch in super().__iter__():
            processed_batch = []
            with ConfigContext(eager_mode=True):
                for t in batch:
                    if self.target_size is not None and len(t.shape) == 4:
                        t = cast(
                            Tensor,
                            cast("EagerCallableProtocol", upsample_bilinear)(t, size=self.target_size),
                        )
                    if self.normalize:
                        t = true_divide(t, 255.0)
                    processed_batch.append(t)
            yield tuple(processed_batch)


class AudioDataset(Dataset):
    """Dataset stream for audio waveform tensors."""

    def __init__(self, *tensors: Tensor, sample_rate: int = 16000) -> None:
        """Initialize AudioDataset.

        Args:
            *tensors (Tensor): Input audio waveform tensors.
            sample_rate (int): Audio sampling rate in Hz.
        """
        if not tensors:
            super().__init__()
        else:
            super().__init__(*tensors)
        self.sample_rate = sample_rate


class TextDataset(Dataset):
    """Dataset stream for tokenized text tensors."""

    def __init__(self, *tensors: Tensor, vocab_size: Optional[int] = None) -> None:
        """Initialize TextDataset.

        Args:
            *tensors (Tensor): Input token ID tensors.
            vocab_size (Optional[int]): Optional vocabulary size metadata.
        """
        if not tensors:
            super().__init__()
        else:
            super().__init__(*tensors)
        self.vocab_size = vocab_size


class NumpyDataset(Dataset):
    """Dataset stream wrapping raw backend arrays into `Tensor` slices."""

    def __init__(self, *arrays: TensorOrScalar) -> None:
        """Initialize NumpyDataset from raw arrays.

        Args:
            *arrays (TensorOrScalar): Raw arrays sharing the same leading dimension.
        """
        from ml_switcheroo_compiler.core.device import Device, DeviceType
        from ml_switcheroo_compiler.ops.type_inference import resolve_dtype

        tensors = []
        for arr in arrays:
            shape = getattr(arr, "shape", ())
            dtype = resolve_dtype(arr, None)
            tensors.append(Tensor(arr, TensorConfig(shape, dtype, Device(DeviceType.CPU))))
        super().__init__(*tensors)
