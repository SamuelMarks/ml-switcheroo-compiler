"""Grain API to ML-Switcheroo-Compiler IR."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Iterable

_T = TypeVar("_T")

# 1. Core Data Structures & Typing


class RecordMetadata:
    """Metadata for a record."""

    def __init__(
        self,
        index: int | None = None,
        record_key: int | None = None,
        rng: object = None,
    ) -> None:
        """Initialize metadata.

        Args:
            index (int | None): The index.
            record_key (int | None): The record_key.
            rng (object): The rng.
        """
        self.index = index
        self.record_key = record_key
        self.rng = rng

    def __str__(self) -> str:
        """String representation.

        Returns:
            str: The computed result.
        """
        return f"RecordMetadata(index={self.index}, record_key={self.record_key})"

    def __eq__(self, other: object) -> bool:
        """Equality check.

        Args:
            other (object): The other.

        Returns:
            bool: The computed result.
        """
        if not isinstance(other, RecordMetadata):
            return False
        return (
            self.index == other.index
            and self.record_key == other.record_key
            and self.rng == other.rng
        )

    def remove_record_key(self) -> None:
        """Remove record key."""
        self.record_key = None


class Record:
    """A record containing metadata and data."""

    def __init__(self, metadata: RecordMetadata | None = None, data: object = None) -> None:
        """Initialize record.

        Args:
            metadata (RecordMetadata | None): The metadata.
            data (object): The data.
        """
        self.metadata = metadata
        self.data = data


class Batch:
    """An alias or representation of batched collections."""


class DatasetOptions:
    """Options for a dataset."""

    def __init__(
        self,
        filter_warn_threshold_ratio: float = 0.1,
        filter_raise_threshold_ratio: float = 0.2,
    ) -> None:
        """Initialize options.

        Args:
            filter_warn_threshold_ratio (float): The filter_warn_threshold_ratio.
            filter_raise_threshold_ratio (float): The filter_raise_threshold_ratio.
        """
        self.filter_warn_threshold_ratio = filter_warn_threshold_ratio
        self.filter_raise_threshold_ratio = filter_raise_threshold_ratio


# 2. Data Sources


class RandomAccessDataSource:
    """A data source allowing random access."""

    def __len__(self) -> int:
        """Get length.

        Returns:
            int: The computed result.
        """
        return 0


class RangeDataSource(RandomAccessDataSource):
    """A data source representing a range of integers."""

    def __init__(self, start: int = 0, stop: int = 0, step: int = 1) -> None:
        """Initialize source.

        Args:
            start (int): The start.
            stop (int): The stop.
            step (int): The step.
        """
        self.start = start
        self.stop = stop
        self.step = step

    def __len__(self) -> int:
        """Get length.

        Returns:
            int: The computed result.
        """
        return max(0, (self.stop - self.start + self.step - 1) // self.step)

    def __getitem__(self, idx: int) -> int:
        """Get item.

        Args:
            idx (int): The idx.

        Returns:
            int: The computed result.
        """
        if idx < 0 or idx >= len(self):
            msg = "Index out of range"
            raise IndexError(msg)
        return self.start + idx * self.step

    def __repr__(self) -> str:
        """Repr.

        Returns:
            str: The computed result.
        """
        return f"RangeDataSource(start={self.start}, stop={self.stop}, step={self.step})"


class InMemoryDataSource(RandomAccessDataSource):
    """An in-memory data source."""

    def __init__(self, elements: list[object] | None = None, name: str | None = None) -> None:
        """Initialize source.

        Args:
            elements (list[object] | None): The elements.
            name (str | None): The name.
        """
        self.elements = elements if elements is not None else []
        self.name = name

    def __len__(self) -> int:
        """Get length.

        Returns:
            int: The computed result.
        """
        return len(self.elements)

    def __getitem__(self, idx: int) -> object:
        """Get item.

        Args:
            idx (int): The idx.

        Returns:
            object: The computed result.
        """
        return self.elements[idx]

    def close(self) -> None:
        """Close data source."""

    def unlink(self) -> None:
        """Unlink data source."""

    def __str__(self) -> str:
        """Str.

        Returns:
            str: The computed result.
        """
        return f"InMemoryDataSource(name={self.name}, len={len(self)})"


class ArrayRecordDataSource(RandomAccessDataSource):
    """A data source for array records."""

    def __init__(self, paths: list[str] | None = None, reader_options: object = None) -> None:
        """Initialize source.

        Args:
            paths (list[str] | None): The paths.
            reader_options (object): The reader_options.
        """
        self.paths = paths if paths is not None else []
        self.reader_options = reader_options


class SharedMemoryDataSource(RandomAccessDataSource):
    """A data source utilizing shared memory."""

    def __init__(self, elements: list[object] | None = None, name: str | None = None) -> None:
        """Initialize source.

        Args:
            elements (list[object] | None): The elements.
            name (str | None): The name.
        """
        self.elements = elements if elements is not None else []
        self.name = name

    def __len__(self) -> int:
        """Get length.

        Returns:
            int: The computed result.
        """
        return len(self.elements)

    def __getitem__(self, idx: int) -> object:
        """Get item.

        Args:
            idx (int): The idx.

        Returns:
            object: The computed result.
        """
        return self.elements[idx]

    def close(self) -> None:
        """Close."""

    def unlink(self) -> None:
        """Unlink."""

    def __str__(self) -> str:
        """Str.

        Returns:
            str: The computed result.
        """
        return f"SharedMemoryDataSource(name={self.name}, len={len(self)})"


# 3. Sampling Strategies & Sharding


class ShardOptions:
    """Options for sharding a dataset."""

    def __init__(
        self,
        shard_index: int = 0,
        shard_count: int = 1,
        drop_remainder: bool = False,
    ) -> None:
        """Initialize sharding options.

        Args:
            shard_index (int): The shard_index.
            shard_count (int): The shard_count.
            drop_remainder (bool): The drop_remainder.
        """
        self.shard_index = shard_index
        self.shard_count = shard_count
        self.drop_remainder = drop_remainder


class NoSharding(ShardOptions):
    """No sharding options."""

    def __init__(
        self,
        shard_index: int = 0,
        shard_count: int = 1,
        drop_remainder: bool = False,
    ) -> None:
        """Initialize no sharding.

        Args:
            shard_index (int): The shard_index.
            shard_count (int): The shard_count.
            drop_remainder (bool): The drop_remainder.
        """
        super().__init__(shard_index, shard_count, drop_remainder)

    def __repr__(self) -> str:
        """Repr.

        Returns:
            str: The computed result.
        """
        return "NoSharding()"


class ShardByJaxProcess(ShardOptions):
    """Sharding options by Jax process."""

    def __init__(self, drop_remainder: bool = False) -> None:
        """Initialize Jax process sharding.

        Args:
            drop_remainder (bool): The drop_remainder.
        """
        super().__init__(shard_index=0, shard_count=1, drop_remainder=drop_remainder)


class sharding:
    """Namespace for sharding options."""

    ShardOptions = ShardOptions
    NoSharding = NoSharding
    ShardByJaxProcess = ShardByJaxProcess


class SequentialSampler:
    """A sampler that produces items sequentially."""

    def __init__(
        self,
        num_records: int = 1,
        shard_options: object | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize sampler.

        Args:
            num_records (int): The num_records.
            shard_options (object | None): The shard_options.
            seed (int | None): The seed.
        """
        self.num_records = num_records
        self.shard_options = shard_options
        self.seed = seed

    def __getitem__(self, idx: int) -> int:
        """Get item.

        Args:
            idx (int): The idx.

        Returns:
            int: The computed result.
        """
        if idx < 0 or idx >= self.num_records:
            msg = "Index out of range"
            raise IndexError(msg)
        return idx

    def __iter__(self) -> Iterable[int]:
        """Iterate.

        Returns:
            Iterable[int]: The computed result.
        """
        return iter(range(self.num_records))

    def __repr__(self) -> str:
        """Repr.

        Returns:
            str: The computed result.
        """
        return f"SequentialSampler(num_records={self.num_records})"


class IndexSampler:
    """A sampler that shuffles indices."""

    def __init__(
        self,
        num_records: int,
        shard_options: object | None = None,
        shuffle: bool = False,
        num_epochs: int = 1,
        seed: int | None = None,
    ) -> None:
        """Initialize sampler.

        Args:
            num_records (int): The num_records.
            shard_options (object | None): The shard_options.
            shuffle (bool): The shuffle.
            num_epochs (int): The num_epochs.
            seed (int | None): The seed.
        """
        self.num_records = num_records
        self.shard_options = shard_options
        self.shuffle = shuffle
        self.num_epochs = num_epochs
        self.seed = seed

    def __getitem__(self, idx: int) -> int:
        """Get item.

        Args:
            idx (int): The idx.

        Returns:
            int: The computed result.
        """
        return idx % self.num_records

    def __iter__(self) -> Iterable[int]:
        """Iterate.

        Returns:
            Iterable[int]: The computed result.
        """
        for _ in range(self.num_epochs):
            yield from range(self.num_records)


Sampler = SequentialSampler

# 4. Operations & Transformations (Compute IR Nodes)


class MapOperation:
    """An operation that maps records."""

    def __init__(self, map_function: Callable[[object], Any] | None = None) -> None:
        """Initialize op.

        Args:
            map_function (Callable[[object], Any] | None): The map_function.
        """
        self.map_function = map_function


class MapWithIndexOperation:
    """An operation that maps records with their index."""

    def __init__(self, map_function: Callable[[int, Any], Any] | None = None) -> None:
        """Initialize op.

        Args:
            map_function (Callable[[int, Any], Any] | None): The map_function.
        """
        self.map_function = map_function


class RandomMapOperation:
    """An operation that maps records randomly."""

    def __init__(self, random_map_function: Callable[[Any, Any], Any] | None = None) -> None:
        """Initialize op.

        Args:
            random_map_function (Callable[[Any, Any], Any] | None): The random_map_function.
        """
        self.random_map_function = random_map_function


class FilterOperation:
    """An operation that filters records."""

    def __init__(self, condition_function: Callable[[object], bool] | None = None) -> None:
        """Initialize op.

        Args:
            condition_function (Callable[[object], bool] | None): The condition_function.
        """
        self.condition_function = condition_function


class FlatMapOperation:
    """An operation that flat-maps records."""

    def __init__(self, map_function: Callable[[object], Iterable[object]] | None = None) -> None:
        """Initialize op.

        Args:
            map_function (Callable[[object], Iterable[object]] | None): The map_function.
        """
        self.map_function = map_function


class BatchOperation:
    """An operation that batches records."""

    def __init__(
        self,
        batch_size: int = 1,
        drop_remainder: bool = False,
        batch_fn: Callable[[list[object]], Any] | None = None,
    ) -> None:
        """Initialize op.

        Args:
            batch_size (int): The batch_size.
            drop_remainder (bool): The drop_remainder.
            batch_fn (Callable[[list[object]], Any] | None): The batch_fn.
        """
        self.batch_size = batch_size
        self.drop_remainder = drop_remainder
        self.batch_fn = batch_fn


def _batch_elements(batch: list[object]) -> object:
    """Batch a list of elements together based on their type.

    Args:
        batch (list[object]): The batch.

    Returns:
        object: The computed result.
    """
    return batch


def batch_and_pad(elements: list[object], batch_size: int) -> object:
    """Batch and pad elements to a given size.

    Args:
        elements (list[object]): The elements.
        batch_size (int): The batch_size.

    Returns:
        object: The computed result.
    """
    pad_len = batch_size - len(elements)
    if pad_len > 0:
        return elements + [None] * pad_len
    return elements


class CopyNumPyArrayToSharedMemoryOperation:
    """An operation that copies NumPy arrays to shared memory."""


class transforms:
    """Namespace for transformations."""

    Filter = FilterOperation
    Map = MapOperation
    MapWithIndex = MapWithIndexOperation
    RandomMap = RandomMapOperation
    Batch = BatchOperation
    FlatMap = FlatMapOperation


Operation = MapOperation

# 5. Execution Engine, Pipeline State & Dataset APIs


class DataLoaderIterator:
    """An iterator for a DataLoader."""

    def __init__(
        self,
        data_loader: DataLoader,
        state: dict[str, Any] | None = None,
        validate_state: bool = True,
    ) -> None:
        """Initialize iterator.

        Args:
            data_loader (DataLoader): The data_loader.
            state (dict[str, Any] | None): The state.
            validate_state (bool): The validate_state.
        """
        self.data_loader = data_loader
        self.state = state if state is not None else {}
        self.validate_state = validate_state

    def get_state(self) -> dict[str, Any]:
        """Get the current state.

        Returns:
            dict[str, Any]: The computed result.
        """
        return self.state

    def set_state(self, state: dict[str, Any]) -> None:
        """Set the current state.

        Args:
            state (dict[str, Any]): The state.
        """
        self.state = state

    def __iter__(self) -> DataLoaderIterator:
        """Return self as iterator.

        Returns:
            DataLoaderIterator: The computed result.
        """
        return self

    def __next__(self) -> object:
        """Get the next element.

        Returns:
            object: The computed result.
        """
        raise StopIteration

    def start_prefetch(self) -> None:
        """Start prefetching."""


class DataLoader:
    """A data loader."""

    def __init__(
        self,
        data_source: object = None,
        sampler: object = None,
        operations: list[object] | None = None,
        worker_count: int = 0,
        worker_buffer_size: int = 1,
        shard_options: object = None,
        read_options: object = None,
        enable_profiling: bool = False,
    ) -> None:
        """Initialize data loader.

        Args:
            data_source (object): The data_source.
            sampler (object): The sampler.
            operations (list[object] | None): The operations.
            worker_count (int): The worker_count.
            worker_buffer_size (int): The worker_buffer_size.
            shard_options (object): The shard_options.
            read_options (object): The read_options.
            enable_profiling (bool): The enable_profiling.
        """
        self.data_source = data_source
        self.sampler = sampler
        self.operations = operations if operations is not None else []
        self.worker_count = worker_count
        self.worker_buffer_size = worker_buffer_size
        self.shard_options = shard_options
        self.read_options = read_options
        self.enable_profiling = enable_profiling

    def __iter__(self) -> DataLoaderIterator:
        """Iterate.

        Returns:
            DataLoaderIterator: The computed result.
        """
        return DataLoaderIterator(self)


class PyGrainCheckpointHandler:
    """A checkpoint handler for PyGrain."""

    def save(self) -> None:
        """Save checkpoint."""

    def restore(self) -> None:
        """Restore checkpoint."""


def assert_equal_output_after_checkpoint(data_loader: object) -> None:
    """Assert equal output after checkpointing.

    Args:
        data_loader (object): The data_loader.
    """


class Dataset(Generic[_T]):
    """A dataset."""

    def __init__(self, elements: list[_T]) -> None:
        """Initialize dataset.

        Args:
            elements (list[_T]): The elements.
        """
        self.elements = elements

    def __len__(self) -> int:
        """Get length.

        Returns:
            int: The computed result.
        """
        return len(self.elements)

    def __getitem__(self, idx: int) -> _T:
        """Get item.

        Args:
            idx (int): The idx.

        Returns:
            _T: The computed result.
        """
        return self.elements[idx]

    def map(self, map_function: Callable[[_T], Any]) -> MapDataset[_T]:
        """Map dataset.

        Args:
            map_function (Callable[[_T], Any]): The map_function.

        Returns:
            MapDataset[_T]: The computed result.
        """
        return MapDataset(self, map_function)

    def map_with_index(self, map_function: Callable[[int, _T], Any]) -> MapDataset[_T]:
        """Map with index.

        Args:
            map_function (Callable[[int, _T], Any]): The map_function.

        Returns:
            MapDataset[_T]: The computed result.
        """
        return MapDataset(self, map_function, with_index=True)

    def filter(self, condition_function: Callable[[_T], bool]) -> FilterDataset[_T]:
        """Filter dataset.

        Args:
            condition_function (Callable[[_T], bool]): The condition_function.

        Returns:
            FilterDataset[_T]: The computed result.
        """
        return FilterDataset(self, condition_function)

    def batch(self, batch_size: int) -> BatchDataset[_T]:
        """Batch dataset.

        Args:
            batch_size (int): The batch_size.

        Returns:
            BatchDataset[_T]: The computed result.
        """
        return BatchDataset(self, batch_size)

    def shuffle(self) -> Dataset[_T]:
        """Shuffle dataset.

        Returns:
            Dataset[_T]: The computed result.
        """
        return self

    def seed(self, seed: int) -> Dataset[_T]:
        """Seed dataset.

        Args:
            seed (int): The seed.

        Returns:
            Dataset[_T]: The computed result.
        """
        return self

    def to_iter_dataset(self) -> IterDataset[_T]:
        """To iter dataset.

        Returns:
            IterDataset[_T]: The computed result.
        """
        return IterDataset(self)

    @classmethod
    def range(cls, *args: int) -> Dataset[int]:
        """Range dataset.

        Args:
            *args: Additional arguments.

        Returns:
            Dataset[int]: The computed result.
        """
        return cls(list(range(*args)))


class MapDataset(Dataset[_T]):
    """A mapped dataset."""

    def __init__(
        self,
        parent: Dataset[_T],
        map_function: Callable[..., Any],
        with_index: bool = False,
    ) -> None:
        """Initialize map dataset.

        Args:
            parent (Dataset[_T]): The parent.
            map_function (Callable[..., Any]): The map_function.
            with_index (bool): The with_index.
        """
        super().__init__([])  # Dummy init
        self.parent = parent
        self.map_function = map_function
        self.with_index = with_index


class FilterDataset(Dataset[_T]):
    """A filtered dataset."""

    def __init__(self, parent: Dataset[_T], condition_function: Callable[[_T], bool]) -> None:
        """Initialize filter dataset.

        Args:
            parent (Dataset[_T]): The parent.
            condition_function (Callable[[_T], bool]): The condition_function.
        """
        super().__init__([])  # Dummy init
        self.parent = parent
        self.condition_function = condition_function


class BatchDataset(Dataset[_T]):
    """A batched dataset."""

    def __init__(self, parent: Dataset[_T], batch_size: int) -> None:
        """Initialize batch dataset.

        Args:
            parent (Dataset[_T]): The parent.
            batch_size (int): The batch_size.
        """
        super().__init__([])  # Dummy init
        self.parent = parent
        self.batch_size = batch_size


class IterDataset(Generic[_T]):
    """An iterative dataset."""

    def __init__(self, dataset: Dataset[_T]) -> None:
        """Initialize iter dataset.

        Args:
            dataset (Dataset[_T]): The dataset.
        """
        self.dataset = dataset
        self._iter = iter(dataset.elements)

    def __iter__(self) -> IterDataset[_T]:
        """Iterate.

        Returns:
            IterDataset[_T]: The computed result.
        """
        return self

    def __next__(self) -> _T:
        """Next.

        Returns:
            _T: The computed result.
        """
        return next(self._iter)


def load(
    source: object,
    num_epochs: int = 1,
    shuffle: bool = False,
    seed: int | None = None,
    shard_options: object | None = None,
    transformations: list[object] | None = None,
    batch_size: int = 1,
    drop_remainder: bool = False,
    worker_count: int = 0,
    read_options: object | None = None,
) -> DataLoader:
    """Load a dataset from a source.

    Args:
        source (object): The source.
        num_epochs (int): The num_epochs.
        shuffle (bool): The shuffle.
        seed (int | None): The seed.
        shard_options (object | None): The shard_options.
        transformations (list[object] | None): The transformations.
        batch_size (int): The batch_size.
        drop_remainder (bool): The drop_remainder.
        worker_count (int): The worker_count.
        read_options (object | None): The read_options.

    Returns:
        DataLoader: The computed result.
    """
    return DataLoader(data_source=source, shard_options=shard_options, read_options=read_options)


def apply_transformations(ds: Dataset[_T], transform: object) -> Dataset[_T]:
    """Apply a transformation to a dataset.

    Args:
        ds (Dataset[_T]): The ds.
        transform (object): The transform.

    Returns:
        Dataset[_T]: The computed result.
    """
    return ds


def get_element_spec(ds: Dataset[_T]) -> object:
    """Get the element spec of a dataset.

    Args:
        ds (Dataset[_T]): The ds.

    Returns:
        object: The computed result.
    """
    return None


class DatasetIterator:
    """Dataset iterator abstraction."""


PyGrainDatasetIterator = DatasetIterator


class DatasetSelectionMap:
    """Dataset Selection Map."""


# 6. Concurrency & Performance Options


class ReadOptions:
    """Read options for a DataLoader."""

    def __init__(self, num_threads: int = 16, prefetch_buffer_size: int = 500) -> None:
        """Initialize read options.

        Args:
            num_threads (int): The num_threads.
            prefetch_buffer_size (int): The prefetch_buffer_size.
        """
        self.num_threads = num_threads
        self.prefetch_buffer_size = prefetch_buffer_size


class MultiprocessingOptions:
    """Multiprocessing options for a DataLoader."""

    def __init__(
        self,
        num_workers: int = 0,
        per_worker_buffer_size: int = 1,
        enable_profiling: bool = False,
    ) -> None:
        """Initialize options.

        Args:
            num_workers (int): The num_workers.
            per_worker_buffer_size (int): The per_worker_buffer_size.
            enable_profiling (bool): The enable_profiling.
        """
        self.num_workers = num_workers
        self.per_worker_buffer_size = per_worker_buffer_size
        self.enable_profiling = enable_profiling


class SharedMemoryArray:
    """A shared memory array."""


class SharedMemoryArrayMetadata:
    """Metadata for a shared memory array."""


class shared_memory_array:
    """Namespace for shared memory array."""

    SharedMemoryArrayMetadata = SharedMemoryArrayMetadata
