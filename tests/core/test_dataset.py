"""Unit tests for core dataset pipeline primitives."""

from __future__ import annotations

import typing

import pytest

from tests.testing_utils import as_callable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def test_dataset_options() -> None:
    """Verify default configuration values on `Options`."""
    from ml_switcheroo_compiler.core.dataset import AutoShardPolicy, Options

    o = Options()
    assert o.autotune_algorithm is None
    assert o.deterministic is None
    assert o.experimental_optimization == {}
    assert o.experimental_distribute == {"auto_shard_policy": AutoShardPolicy.AUTO}
    assert o.threading == {}


def test_dataset_init() -> None:
    """Verify `Dataset` initialization, leading dimension validation, and `from_tensor_slices`."""
    import numpy as np

    from ml_switcheroo_compiler.core.dataset import Dataset
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t1 = Tensor(np.array([1, 2]), TensorConfig(shape=(2,), dtype="float32", device="cpu"))
    t2 = Tensor(np.array([3, 4]), TensorConfig(shape=(2,), dtype="float32", device="cpu"))

    ds = Dataset(t1, t2)
    assert ds.length == 2

    t3 = Tensor(np.array([1, 2, 3]), TensorConfig(shape=(3,), dtype="float32", device="cpu"))
    with pytest.raises(ValueError, match="same leading dimension"):
        Dataset(t1, t3)

    ds_empty = Dataset()
    assert ds_empty.length == 0

    ds_from = Dataset.from_tensor_slices(t1, t2)
    assert ds_from.length == 2


def test_dataset_from_tensor_slices_error() -> None:
    """Verify `Dataset.from_tensor_slices()` raises `ValueError` when called with no tensors."""
    from ml_switcheroo_compiler.core.dataset import Dataset

    with pytest.raises(ValueError, match="At least one tensor"):
        Dataset.from_tensor_slices()


def test_dataset_options_batch_unbatch_map_batch() -> None:
    """Verify batching, unbatching, ragged batching, and windowed dataset stream methods."""
    import numpy as np

    from ml_switcheroo_compiler.core.dataset import Dataset, Options
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t1 = Tensor(np.array([1, 2]), TensorConfig(shape=(2,), dtype="float32", device="cpu"))
    ds = Dataset(t1)

    ds_from_gen = Dataset.from_list([1, 2])
    assert ds_from_gen.length == 2

    ds = ds.with_options(Options())

    with pytest.raises(ValueError, match="batch_size must be positive"):
        ds.batch(0)

    ds = ds.batch(2, True)
    assert ds._batch_size == 2
    assert ds._drop_remainder

    ds = ds.unbatch()
    assert ds._batch_size == 1

    ds = ds.pad_to_cardinality(2)

    with pytest.raises(ValueError, match="batch_size must be positive"):
        ds.dense_to_ragged_batch(0)

    ds = ds.dense_to_ragged_batch(2, drop_remainder=True)
    assert ds._batch_size == 2

    ds = ds.map_and_batch(lambda x: x, 2, drop_remainder=True)
    assert ds._batch_size == 2

    ds = ds.group_by_window(lambda x: x, lambda x: x, 2)
    ds = ds.rejection_resample(lambda x: x, [0.5, 0.5])
    assert ds._batch_size == 2


def test_dataset_iter_save_load_iters() -> None:
    """Verify `Dataset` iteration, shuffling, mapping, save/load, and iterator wrappers."""
    import numpy as np

    from ml_switcheroo_compiler.core.dataset import ArrayIterator, Dataset, NumpyIterator
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t1 = Tensor(np.array([1, 2, 3]), TensorConfig(shape=(3,), dtype="int32", device="cpu"))
    ds = Dataset(t1)

    ds = ds.batch(2, drop_remainder=False)
    batches = list(ds)
    assert len(batches) == 2
    assert as_callable(len)(batches[0][0].data) == 2
    assert as_callable(len)(batches[1][0].data) == 1

    ds = ds.batch(2, drop_remainder=True)
    batches = list(ds)
    assert len(batches) == 1

    ds._map_func = lambda x: x
    batches = list(ds)
    assert len(batches) == 1

    ds = ds.shuffle(buffer_size=2, seed=42)
    batches = list(ds)
    assert len(batches) == 1

    ds_empty = Dataset()
    assert len(list(ds_empty)) == 1

    ds_save = Dataset()
    assert ds_save.save("dummy") is ds_save
    assert isinstance(Dataset.load("dummy"), Dataset)

    it = NumpyIterator(ds)
    assert iter(it) is it
    n = next(it)
    assert as_callable(len)(n) == 2

    ds2 = Dataset(t1)
    it2 = ArrayIterator(ds2)
    assert iter(it2) is it2
    n2 = next(it2)
    assert as_callable(len)(n2) == 1


def test_dataset_coverage_remaining() -> None:
    """Verify specialized dataset subclasses and invalid shuffle/prefetch buffer sizes."""
    import numpy as np

    from ml_switcheroo_compiler.core.dataset import (
        AudioDataset,
        CsvDataset,
        Dataset,
        ImageDataset,
        NumpyDataset,
        SqlDataset,
        TextDataset,
    )
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t1 = Tensor(np.array([1, 2]), TensorConfig(shape=(2,), dtype="float32", device="cpu"))
    ds = Dataset(t1)

    ds = ds.parallel_interleave(lambda x: x, cycle_length=1)
    ds = ds.prefetch_to_device("cpu")

    with pytest.raises(ValueError, match="buffer_size must be positive"):
        ds.shuffle(-1)

    with pytest.raises(ValueError, match="buffer_size must be positive"):
        ds.prefetch(-1)

    ds = ds.prefetch(1)
    assert ds._prefetch_buffer == 1

    csv_ds = CsvDataset("dummy.csv")
    assert csv_ds.filenames == "dummy.csv"
    SqlDataset("dummy.db", "SELECT 1", "SELECT 1", (DType.Float32,))
    AudioDataset(t1)
    AudioDataset()

    ImageDataset(t1, target_size=(2, 2), normalize=True)
    ImageDataset(target_size=(2, 2), normalize=True)

    TextDataset(t1, vocab_size=10)
    TextDataset(vocab_size=10)

    NumpyDataset(np.array([1, 2]))

    img_t = Tensor(np.ones((1, 2, 2, 1)), TensorConfig(shape=(1, 2, 2, 1), dtype="float32", device="cpu"))
    img_ds3 = ImageDataset(img_t, target_size=(4, 4), normalize=True)

    class FakeUpsample:
        """Stub bilinear upsample callable."""

        def __call__(self, t: EagerValue, size: EagerValue) -> EagerValue:
            """Return input tensor unchanged.

            Args:
                t (EagerValue): Input tensor.
                size (EagerValue): Target spatial size.

            Returns:
                EagerValue: Unmodified tensor `t`.
            """
            _ = size
            return t

    class FakeTrueDiv:
        """Stub division callable."""

        def __call__(self, t: EagerValue, v: EagerValue) -> EagerValue:
            """Return input tensor unchanged.

            Args:
                t (EagerValue): Dividend tensor.
                v (EagerValue): Divisor scalar.

            Returns:
                EagerValue: Unmodified tensor `t`.
            """
            _ = v
            return t

    from unittest.mock import patch

    import ml_switcheroo_compiler.ops.binary as binary_ops
    import ml_switcheroo_compiler.ops.nn.upsample_ops as upsample_ops

    with patch.object(upsample_ops, "upsample_bilinear", FakeUpsample()):
        with patch.object(binary_ops, "true_divide", FakeTrueDiv()):
            batches = list(img_ds3)
            assert len(batches) == 1


def test_dataset_coverage_remaining_2() -> None:
    """Verify snapshot, `BinaryRecordReader`, unseeded shuffle, and `ImageDataset` branch variants."""
    import numpy as np

    from ml_switcheroo_compiler.core.dataset import BinaryRecordReader, Dataset, ImageDataset, TFRecordReader
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t1 = Tensor(np.array([1, 2]), TensorConfig(shape=(2,), dtype="float32", device="cpu"))
    ds = Dataset(t1)

    ds = ds.snapshot("dummy")

    reader = TFRecordReader("dummy.tfrecord")
    assert isinstance(reader, BinaryRecordReader)

    ds = ds.shuffle(buffer_size=2, seed=None)
    assert len(list(ds)) == 2

    img_t = Tensor(np.ones((1, 2, 2, 1)), TensorConfig(shape=(1, 2, 2, 1), dtype="float32", device="cpu"))
    img_ds4 = ImageDataset(img_t, target_size=None, normalize=True)
    assert len(list(img_ds4)) == 1

    img_ds5 = ImageDataset(img_t, target_size=(4, 4), normalize=False)

    class FakeUpsample:
        """Stub bilinear upsample callable."""

        def __call__(self, t: EagerValue, size: EagerValue) -> EagerValue:
            """Return input tensor unchanged.

            Args:
                t (EagerValue): Input tensor.
                size (EagerValue): Target spatial size.

            Returns:
                EagerValue: Unmodified tensor `t`.
            """
            _ = size
            return t

    from unittest.mock import patch

    import ml_switcheroo_compiler.ops.nn.upsample_ops as upsample_ops

    with patch.object(upsample_ops, "upsample_bilinear", FakeUpsample()):
        batches5 = list(img_ds5)
    assert len(batches5) == 1
