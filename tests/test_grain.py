"""Tests for the grain module."""

import pytest

from ml_switcheroo_compiler.grain import (
    ArrayRecordDataSource,
    Batch,
    BatchOperation,
    CopyNumPyArrayToSharedMemoryOperation,
    DataLoader,
    Dataset,
    DatasetIterator,
    DatasetOptions,
    DatasetSelectionMap,
    FilterOperation,
    FlatMapOperation,
    IndexSampler,
    InMemoryDataSource,
    MapOperation,
    MapWithIndexOperation,
    MultiprocessingOptions,
    NoSharding,
    Operation,
    PyGrainCheckpointHandler,
    PyGrainDatasetIterator,
    RandomMapOperation,
    RangeDataSource,
    ReadOptions,
    Record,
    RecordMetadata,
    Sampler,
    SequentialSampler,
    ShardByJaxProcess,
    ShardOptions,
    SharedMemoryArray,
    SharedMemoryArrayMetadata,
    SharedMemoryDataSource,
    _batch_elements,
    apply_transformations,
    assert_equal_output_after_checkpoint,
    batch_and_pad,
    get_element_spec,
    load,
    sharding,
    shared_memory_array,
    transforms,
)


def test_record_metadata() -> None:
    """Test RecordMetadata."""
    rm1 = RecordMetadata(index=1, record_key=10, rng="seed")
    rm2 = RecordMetadata(index=1, record_key=10, rng="seed")
    rm3 = RecordMetadata(index=2)
    assert rm1 == rm2
    assert rm1 != rm3
    assert rm1 != "not metadata"
    assert str(rm1) == "RecordMetadata(index=1, record_key=10)"
    rm1.remove_record_key()
    assert rm1.record_key is None


def test_record() -> None:
    """Test Record."""
    rm = RecordMetadata(index=1)
    rec = Record(metadata=rm, data={"a": 1})
    assert rec.metadata == rm
    assert rec.data == {"a": 1}


def test_batch() -> None:
    """Test Batch."""
    b = Batch()
    assert b is not None


def test_dataset_options() -> None:
    """Test DatasetOptions."""
    opt = DatasetOptions(filter_warn_threshold_ratio=0.5, filter_raise_threshold_ratio=0.8)
    assert opt.filter_warn_threshold_ratio == 0.5
    assert opt.filter_raise_threshold_ratio == 0.8


def test_range_data_source() -> None:
    """Test RangeDataSource."""
    rds = RangeDataSource(0, 10, 2)
    assert len(rds) == 5
    assert rds[0] == 0
    assert rds[4] == 8
    with pytest.raises(IndexError):
        _ = rds[5]
    with pytest.raises(IndexError):
        _ = rds[-1]
    assert repr(rds) == "RangeDataSource(start=0, stop=10, step=2)"


def test_in_memory_data_source() -> None:
    """Test InMemoryDataSource."""
    ims = InMemoryDataSource([1, 2, 3], name="test")
    assert len(ims) == 3
    assert ims[1] == 2
    ims.close()
    ims.unlink()
    assert str(ims) == "InMemoryDataSource(name=test, len=3)"


def test_array_record_data_source() -> None:
    """Test ArrayRecordDataSource."""
    ards = ArrayRecordDataSource(["a", "b"], reader_options={"a": 1})
    assert len(ards) == 0


def test_shared_memory_data_source() -> None:
    """Test SharedMemoryDataSource."""
    smds = SharedMemoryDataSource([4, 5], name="shared")
    assert len(smds) == 2
    assert smds[0] == 4
    smds.close()
    smds.unlink()
    assert str(smds) == "SharedMemoryDataSource(name=shared, len=2)"


def test_sharding() -> None:
    """Test ShardOptions, NoSharding, ShardByJaxProcess."""
    so = ShardOptions(shard_index=1, shard_count=2, drop_remainder=True)
    assert so.shard_index == 1
    ns = NoSharding()
    assert repr(ns) == "NoSharding()"
    sb = ShardByJaxProcess(drop_remainder=True)
    assert sb.drop_remainder is True
    assert sharding.ShardOptions == ShardOptions


def test_samplers() -> None:
    """Test SequentialSampler and IndexSampler."""
    ss = SequentialSampler(num_records=2)
    assert ss[0] == 0
    with pytest.raises(IndexError):
        _ = ss[3]
    with pytest.raises(IndexError):
        _ = ss[-1]
    assert list(ss) == [0, 1]
    assert repr(ss) == "SequentialSampler(num_records=2)"

    iss = IndexSampler(num_records=3, num_epochs=2)
    assert iss[4] == 1
    assert len(list(iss)) == 6
    assert Sampler == SequentialSampler


def test_operations() -> None:
    """Test operations and transformations."""
    _ = MapOperation(lambda x: x)
    _ = MapWithIndexOperation(lambda i, x: x)
    _ = RandomMapOperation(lambda x, rng: x)
    _ = FilterOperation(lambda x: True)
    _ = FlatMapOperation(lambda x: [x])
    _ = BatchOperation(batch_size=2)
    assert _batch_elements([1, 2]) == [1, 2]
    assert batch_and_pad([1], 3) == [1, None, None]
    assert batch_and_pad([1, 2], 2) == [1, 2]
    assert transforms.Map == MapOperation
    assert Operation == MapOperation


def test_data_loader() -> None:
    """Test DataLoader."""
    dl = DataLoader(data_source=InMemoryDataSource([1, 2]))
    it = iter(dl)
    it.set_state({"a": 1})
    assert it.get_state() == {"a": 1}
    it.start_prefetch()


def test_checkpoint_handler() -> None:
    """Test PyGrainCheckpointHandler."""
    ch = PyGrainCheckpointHandler()
    ch.save()
    ch.restore()
    assert_equal_output_after_checkpoint(None)


def test_datasets() -> None:
    """Test Dataset."""
    ds = Dataset.range(5)
    assert len(ds) == 5
    assert ds[1] == 1

    md = ds.map(lambda x: x * 2)
    assert md.with_index is False
    mwi = ds.map_with_index(lambda i, x: x)
    assert mwi.with_index is True

    fd = ds.filter(lambda x: x > 2)
    assert fd.condition_function(3) is True

    bd = ds.batch(2)
    assert bd.batch_size == 2

    assert ds.shuffle() is ds
    assert ds.seed(1) is ds

    id_ds = ds.to_iter_dataset()
    assert next(id_ds) == 0
    assert next(id_ds) == 1
    assert list(id_ds) == [2, 3, 4]


def test_utils() -> None:
    """Test utils."""
    assert apply_transformations(Dataset.range(1), None).elements == [0]
    assert get_element_spec(Dataset.range(1)) is None


def test_misc() -> None:
    """Test misc classes."""
    dl = load(InMemoryDataSource([1]))
    assert dl.data_source.elements == [1]
    ReadOptions()
    MultiprocessingOptions()
    SharedMemoryArray()
    SharedMemoryArrayMetadata()
    assert shared_memory_array.SharedMemoryArrayMetadata == SharedMemoryArrayMetadata
    DatasetIterator()
    PyGrainDatasetIterator()
    DatasetSelectionMap()
    CopyNumPyArrayToSharedMemoryOperation()


def test_data_loader_iterator_iteration() -> None:
    """Test DataLoaderIterator."""
    dl = DataLoader(data_source=InMemoryDataSource([1, 2]))
    it = iter(dl)
    assert iter(it) is it
    with pytest.raises(StopIteration):
        next(it)
