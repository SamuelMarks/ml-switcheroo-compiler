# Grain API to ML-Switcheroo-Compiler IR Todo Plan

This document outlines an exhaustive list of features, classes, methods, and semantic behaviors that `ml-switcheroo-compiler` needs to implement. It includes intermediate representation (IR) nodes, compiler passes, execution backend functionality, namespace aliases, and iterative dataset wrappers to fully support the frontend definitions in `zero-grain` and guarantee 100% syntactic and semantic parity to Google's Grain framework.

## 1. Core Data Structures & Typing

The fundamental units of data flowing through the compiler's IR graph.

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
| :--- | :--- | :--- | :--- | :--- |
| - [ ] | `RecordMetadata` | `class RecordMetadata(index: Optional[int] = None, record_key: Optional[int] = None, rng: Any = None)` | Metadata for a record. | Must track `index`, `record_key`, and `rng` seeds per record within the graph. Important for determinism. Provides `__str__`, `__eq__`, and `remove_record_key`. |
| - [ ] | `Record` | `class Record(metadata: Optional[RecordMetadata] = None, data: Any = None)` | A record containing metadata and data. | Encapsulates `RecordMetadata` alongside `data` (tensors, scalars, and PyTrees) traversing the pipeline. |
| - [ ] | `Batch` | `class Batch()` | An alias or representation of batched collections. | Representation of batched collections in the compiler IR. Requires shape inference constraints. |
| - [ ] | `DatasetOptions` | `class DatasetOptions(filter_warn_threshold_ratio: float = 0.1, filter_raise_threshold_ratio: float = 0.2)` | Options for a dataset. | Configuration object impacting runtime fault tolerance. |

## 2. Data Sources (Input IR Nodes)

Root nodes in the execution graph that ingest data into the pipeline.

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
| :--- | :--- | :--- | :--- | :--- |
| - [ ] | `RangeDataSource` | `class RangeDataSource(start: int = 0, stop: int = 0, step: int = 1)` | A data source representing a range of integers. | Generate range-based index streams purely in IR without hitting host memory. Implements `__len__`, `__getitem__`, `__repr__`. |
| - [ ] | `InMemoryDataSource` | `class InMemoryDataSource(elements: Optional[List[Any]] = None, name: Optional[str] = None)` | An in-memory data source. | Embed statically known elements into the compiled graph (or feed via optimized host-to-device streaming). Implements `__len__`, `__getitem__`, `close`, `unlink`, `__str__`. |
| - [ ] | `ArrayRecordDataSource` | `class ArrayRecordDataSource(paths: Optional[List[str]] = None, reader_options: Any = None)` | A data source for array records. | High-performance binary file reading node. Needs Mock/fallback when compiled without C++ extensions. Inherits from `RandomAccessDataSource`. |
| - [ ] | `SharedMemoryDataSource` | `class SharedMemoryDataSource(elements: Optional[List[Any]] = None, name: Optional[str] = None)` | A data source utilizing shared memory. | IR node reading IPC memory pools for multi-processing. Implements `__len__`, `__getitem__`, `close`, `unlink`, `__str__`. |
| - [ ] | `RandomAccessDataSource` | `class RandomAccessDataSource()` | A data source allowing random access. | Abstract base class tracking indexing guarantees and lengths in type constraints. Implements `__len__`. |

## 3. Sampling Strategies & Sharding

Nodes responsible for sequence index generation and distributed parallel execution chunking.

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
| :--- | :--- | :--- | :--- | :--- |
| - [ ] | `SequentialSampler` | `class SequentialSampler(num_records: int = 1, shard_options: Optional[Any] = None, seed: Optional[int] = None)` | A sampler that produces items sequentially. | Linear incrementing sampler node; simple counter loop. Implements `__getitem__`, `__iter__`, `__repr__`. |
| - [ ] | `IndexSampler` | `class IndexSampler(num_records: int, shard_options: Optional[Any] = None, shuffle: bool = False, num_epochs: int = 1, seed: Optional[int] = None)` | A sampler that shuffles indices. | Epoch-aware, seeded, optionally shuffled sampler. Needs reproducible `numpy` style RNG semantics mapped to compiler primitives. Implements `__getitem__`, `__iter__`. |
| - [ ] | `Sampler` | `Sampler = SequentialSampler` | Alias for Sampler. | Provide backward compatibility alias. |
| - [ ] | `ShardOptions` | `class ShardOptions(shard_index: int = 0, shard_count: int = 1, drop_remainder: bool = False)` | Options for sharding a dataset. | Base sharding logic. Support for partial evaluations per-shard. |
| - [ ] | `NoSharding` | `class NoSharding(shard_index: int = 0, shard_count: int = 1, drop_remainder: bool = False)` | No sharding options. | Single shard execution model. Implements `__repr__`. |
| - [ ] | `ShardByJaxProcess` | `class ShardByJaxProcess(drop_remainder: bool = False)` | Sharding options by Jax process. | Automatic retrieval of distributed rank logic and multi-process index assignment mapping. |
| - [ ] | `sharding` namespace | `class sharding` | Namespace for sharding options. | Contains `ShardOptions`, `NoSharding`, `ShardByJaxProcess`. |

## 4. Operations & Transformations (Compute IR Nodes)

Nodes mapping, filtering, or altering the batch dimensions of data.

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
| :--- | :--- | :--- | :--- | :--- |
| - [ ] | `MapOperation` | `class MapOperation(map_function: Optional[Callable[[Any], Any]] = None)` | An operation that maps records. | Execute stateless scalar/PyTree mapping functions. Includes `Map` and `MapTransform` aliases. |
| - [ ] | `MapWithIndexOperation` | `class MapWithIndexOperation(map_function: Optional[Callable[[int, Any], Any]] = None)` | An operation that maps records with their index. | Mapping that injects the current sequence index (enumerate behavior). Includes `MapWithIndex` and `MapWithIndexTransform` aliases. |
| - [ ] | `RandomMapOperation` | `class RandomMapOperation(random_map_function: Optional[Callable[[Any, Any], Any]] = None)` | An operation that maps records randomly. | Deterministically seeded random mapping (JAX-style PRNG passing threading in IR). Includes `RandomMap`, `RandomMapTransform`, and `TfRandomMap` aliases. |
| - [ ] | `FilterOperation` | `class FilterOperation(condition_function: Optional[Callable[[Any], bool]] = None)` | An operation that filters records. | Dynamic control-flow / predication node to selectively drop elements. Includes `Filter` and `FilterTransform` aliases. |
| - [ ] | `FlatMapOperation` | `class FlatMapOperation(map_function: Optional[Callable[[Any], Iterable[Any]]] = None)` | An operation that flat-maps records. | Sequence unrolling/flattening logic in the IR. Includes `FlatMap` alias. |
| - [ ] | `BatchOperation` | `class BatchOperation(batch_size: int = 1, drop_remainder: bool = False, batch_fn: Optional[Callable[[List[Any]], Any]] = None)` | An operation that batches records. | Tree-aware axis concatenation node. Needs tensor reshape/cat ops. Includes `Batch` alias. |
| - [ ] | `_batch_elements` | `def _batch_elements(batch: List[Any]) -> Any` | Batch a list of elements together based on their type. | PyTree Batching: Recursive batching across `dict`, `tuple`, `list`, `NamedTuple`, and `@dataclass` structures. |
| - [ ] | `batch_and_pad` | `def batch_and_pad(elements: List[Any], batch_size: int) -> Any` | Batch and pad elements to a given size. | Required for fixed-shape tensor constraints in compiler IR when `drop_remainder` is False. |
| - [ ] | `CopyNumPyArrayTo...`| `class CopyNumPyArrayToSharedMemoryOperation()` | An operation that copies NumPy arrays to shared memory. | Host to IPC memory movement optimizations. Includes `CopyNumPyArrayToSharedMemory` alias. |
| - [ ] | `transforms` namespace | `class transforms` | Namespace for transformations. | Contains `Filter`, `Map`, `MapWithIndex`, `RandomMap`, `Batch`, `FlatMap`. |
| - [ ] | `Operation` | `Operation = MapOperation` | Base interface alias | Make sure type hints resolve correctly. |

## 5. Execution Engine, Pipeline State & Dataset APIs

Classes dictating the runtime execution of the compiled graph and fluent builder APIs.

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
| :--- | :--- | :--- | :--- | :--- |
| - [ ] | `DataLoader` | `class DataLoader(data_source: Any = None, sampler: Any = None, operations: Optional[List[Any]] = None, worker_count: int = 0, worker_buffer_size: int = 1, shard_options: Any = None, read_options: Any = None, enable_profiling: bool = False)` | A data loader. | Compile down the source -> sampler -> transforms pipeline into an execution plan. Returns `DataLoaderIterator`. |
| - [ ] | `DataLoaderIterator` | `class DataLoaderIterator(data_loader: "DataLoader", state: Optional[Dict[str, Any]] = None, validate_state: bool = True)` | An iterator for a DataLoader. | State machine logic for pipeline traversal. Includes `get_state` / `set_state` for checkpointing mid-epoch deterministically, and `start_prefetch()`. |
| - [ ] | `PyGrainCheckpointHandler`| `class PyGrainCheckpointHandler()` | A checkpoint handler for PyGrain. | High-level integration to store iterator snapshots efficiently. Includes `CheckpointHandler` alias. Provides `save` and `restore` stubs. |
| - [ ] | `assert_equal_output_after_checkpoint` | `def assert_equal_output_after_checkpoint(data_loader: Any) -> None` | Assert equal output after checkpointing. | Test utility that needs implementing to verify deterministic state resumption. |
| - [ ] | `Dataset` | `class Dataset(Generic[_T])` | A dataset. | Fluent API Builder Parity: Implement `.map()`, `.map_with_index()`, `.filter()`, `.batch()`, `.shuffle()`, `.seed()`, `.to_iter_dataset()`, and `@classmethod range()`. Must support `__len__` and `__getitem__`. |
| - [ ] | `MapDataset` | `class MapDataset(Dataset[_T])` | A mapped dataset. | Concrete type for dataset mutations. |
| - [ ] | `FilterDataset` | `class FilterDataset(Dataset[_T])` | A filtered dataset. | Concrete type for dataset mutations. |
| - [ ] | `BatchDataset` | `class BatchDataset(Dataset[_T])` | A batched dataset. | Concrete type for dataset mutations. |
| - [ ] | `IterDataset` | `class IterDataset(Generic[_T])` | An iterative dataset. | Wrapping `Dataset` configuration into a runtime `DataLoader` and providing `__iter__` / `__next__`. |
| - [ ] | `load` | `def load(source: Any, num_epochs: int = 1, shuffle: bool = False, seed: Optional[int] = None, shard_options: Optional[Any] = None, transformations: Optional[List[Any]] = None, batch_size: int = 1, drop_remainder: bool = False, worker_count: int = 0, read_options: Optional[Any] = None) -> DataLoader` | Load a dataset from a source. | Helper to translate `load()` invocations directly into a pipeline initialization pass. |
| - [ ] | `apply_transformations` | `def apply_transformations(ds: Dataset[_T], transform: Any) -> Dataset[_T]` | Apply a transformation to a dataset. | Utility function. |
| - [ ] | `get_element_spec` | `def get_element_spec(ds: Dataset[_T]) -> Any` | Get the element spec of a dataset. | Critical compiler pass needed to infer structure/shapes prior to runtime. |
| - [ ] | `DatasetIterator` / `PyGrainDatasetIterator` | `class DatasetIterator` | Iterator abstractions | Aliases to ensure type annotations and `isinstance` checks work. |
| - [ ] | `DatasetSelectionMap` | `class DatasetSelectionMap` | Dataset Selection Map | Type stub alias. |

## 6. Concurrency & Performance Options

Settings for threading, processing, and caching memory.

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
| :--- | :--- | :--- | :--- | :--- |
| - [ ] | `ReadOptions` | `class ReadOptions(num_threads: int = 16, prefetch_buffer_size: int = 500)` | Read options for a DataLoader. | Parameterizing async I/O worker constraints. |
| - [ ] | `MultiprocessingOptions`| `class MultiprocessingOptions(num_workers: int = 0, per_worker_buffer_size: int = 1, enable_profiling: bool = False)` | Multiprocessing options for a DataLoader. | Handle worker pool allocation in the backend. Support process crashes securely. |
| - [ ] | `SharedMemoryArray` | `class SharedMemoryArray` | A shared memory array. | Memory layout descriptor. |
| - [ ] | `SharedMemoryArrayMetadata`| `class SharedMemoryArrayMetadata` | Metadata for a shared memory array. | Layout offsets/sizes tracking. |
| - [ ] | `shared_memory_array` namespace| `class shared_memory_array` | Namespace | Contains `SharedMemoryArrayMetadata`. |
