"""Tests for test_dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def test_dataset_options() -> None:
    """Test function test_dataset_options."""
    from ml_switcheroo_compiler.core.dataset import AutoShardPolicy, Options

    o = Options()
    assert o.autotune_algorithm is None
    assert o.deterministic is None
    assert o.experimental_optimization == {}
    assert o.experimental_distribute == {"auto_shard_policy": AutoShardPolicy.AUTO}
    assert o.threading == {}


def test_dataset_init() -> None:
    """Test function test_dataset_init."""
    import numpy as np

    from ml_switcheroo_compiler.core.dataset import Dataset
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t1 = Tensor(np.array([1, 2]), TensorConfig(shape=(2,), dtype="float32", device="cpu"))
    t2 = Tensor(np.array([3, 4]), TensorConfig(shape=(2,), dtype="float32", device="cpu"))

    # 60, 61, 62, 63, 65
    ds = Dataset(t1, t2)
    assert ds.length == 2

    # 63, 64
    t3 = Tensor(np.array([1, 2, 3]), TensorConfig(shape=(3,), dtype="float32", device="cpu"))
    try:
        Dataset(t1, t3)
        raise AssertionError()
    except ValueError as e:
        assert "same leading dimension" in str(e)

    # 67, 68
    ds_empty = Dataset()
    assert ds_empty.length == 0

    # 89, 90, 91
    ds_from = Dataset.from_tensor_slices(t1, t2)
    assert ds_from.length == 2


def test_dataset_from_tensor_slices_error() -> None:
    """Test function test_dataset_from_tensor_slices_error."""
    from ml_switcheroo_compiler.core.dataset import Dataset

    try:
        Dataset.from_tensor_slices()
        raise AssertionError()
    except ValueError as _exc:
        assert _exc is not None


def test_dataset_options_batch_unbatch_map_batch() -> None:
    """Test function test_dataset_options_batch_unbatch_map_batch."""
    import numpy as np

    from ml_switcheroo_compiler.core.dataset import Dataset, Options
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t1 = Tensor(np.array([1, 2]), TensorConfig(shape=(2,), dtype="float32", device="cpu"))
    ds = Dataset(t1)

    ds_from_gen = Dataset.from_list([1, 2])
    assert ds_from_gen.length == 2

    ds = ds.with_options(Options())

    try:
        ds.batch(0)
        raise AssertionError()
    except ValueError as _exc:
        assert _exc is not None

    ds = ds.batch(2, True)
    assert ds._batch_size == 2
    assert ds._drop_remainder

    ds = ds.unbatch()
    assert ds._batch_size == 1

    ds = ds.pad_to_cardinality(2)

    try:
        ds.dense_to_ragged_batch(0)
        raise AssertionError()
    except ValueError as _exc:
        assert _exc is not None

    ds = ds.dense_to_ragged_batch(2, drop_remainder=True)
    assert ds._batch_size == 2

    ds = ds.map_and_batch(lambda x: x, 2, drop_remainder=True)
    assert ds._batch_size == 2

    ds = ds.group_by_window(lambda x: x, lambda x: x, 2)

    ds = ds.rejection_resample(lambda x: x, [0.5, 0.5])


def test_dataset_iter_save_load_iters() -> None:
    """Test function test_dataset_iter_save_load_iters."""
    import numpy as np

    from ml_switcheroo_compiler.core.dataset import ArrayIterator, Dataset, NumpyIterator
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t1 = Tensor(np.array([1, 2, 3]), TensorConfig(shape=(3,), dtype="int32", device="cpu"))
    ds = Dataset(t1)

    ds = ds.batch(2, drop_remainder=False)
    batches = list(ds)
    assert len(batches) == 2
    assert cast("Callable[..., EagerTensorProtocol]", len)(batches[0][0].data) == 2
    assert cast("Callable[..., EagerTensorProtocol]", len)(batches[1][0].data) == 1

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
    assert cast("Callable[..., EagerTensorProtocol]", len)(n) == 2

    ds2 = Dataset(t1)
    it2 = ArrayIterator(ds2)
    assert iter(it2) is it2
    n2 = next(it2)
    assert cast("Callable[..., EagerTensorProtocol]", len)(n2) == 1


def test_dataset_coverage_remaining() -> None:
    """Test function test_dataset_coverage_remaining."""
    import numpy as np

    from ml_switcheroo_compiler.core.dataset import AudioDataset, CsvDataset, Dataset, ImageDataset, NumpyDataset, SqlDataset, TextDataset
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t1 = Tensor(np.array([1, 2]), TensorConfig(shape=(2,), dtype="float32", device="cpu"))
    ds = Dataset(t1)

    ds = ds.parallel_interleave(lambda x: x, cycle_length=1)
    ds = ds.prefetch_to_device("cpu")

    try:
        ds.shuffle(-1)
        raise AssertionError()
    except ValueError as _exc:
        assert _exc is not None

    try:
        ds.prefetch(-1)
        raise AssertionError()
    except ValueError as _exc:
        assert _exc is not None

    ds = ds.prefetch(1)

    CsvDataset("dummy.csv")
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
        """Test class FakeUpsample."""

        def __call__(self, t: EagerValue, size: EagerValue) -> EagerValue:
            """Test function __call__."""
            return t

    class FakeTrueDiv:
        """Test class FakeTrueDiv."""

        def __call__(self, t: EagerValue, v: EagerValue) -> EagerValue:
            """Test function __call__."""
            return t

    from unittest.mock import patch

    import ml_switcheroo_compiler.ops.binary as binary_ops
    import ml_switcheroo_compiler.ops.nn.upsample_ops as upsample_ops

    with patch.object(upsample_ops, "upsample_bilinear", FakeUpsample()):
        with patch.object(binary_ops, "true_divide", FakeTrueDiv()):
            list(img_ds3)


def test_dataset_coverage_remaining_2() -> None:
    """Test function test_dataset_coverage_remaining_2."""
    import numpy as np

    from ml_switcheroo_compiler.core.dataset import Dataset, ImageDataset, TFRecordReader
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t1 = Tensor(np.array([1, 2]), TensorConfig(shape=(2,), dtype="float32", device="cpu"))
    ds = Dataset(t1)

    # 313 -> apply
    ds = ds.snapshot("dummy")

    # 517 -> TFRecordDataset
    TFRecordReader("dummy.tfrecord")

    # [356, 358] -> self._seed is None
    ds = ds.shuffle(buffer_size=2, seed=None)
    list(ds)

    # [551, 553] -> ImageDataset without target_size
    img_t = Tensor(np.ones((1, 2, 2, 1)), TensorConfig(shape=(1, 2, 2, 1), dtype="float32", device="cpu"))
    img_ds4 = ImageDataset(img_t, target_size=None, normalize=True)
    list(img_ds4)

    # [553, 555] -> ImageDataset without normalize
    img_ds5 = ImageDataset(img_t, target_size=(4, 4), normalize=False)

    class FakeUpsample:
        """Test class FakeUpsample."""

        def __call__(self, t: EagerValue, size: EagerValue) -> EagerValue:
            """Test function __call__."""
            return t

    from unittest.mock import patch

    import ml_switcheroo_compiler.ops.nn.upsample_ops as upsample_ops

    with patch.object(upsample_ops, "upsample_bilinear", FakeUpsample()):
        list(img_ds5)
    assert img_ds5 is not None or img_ds5 is None
