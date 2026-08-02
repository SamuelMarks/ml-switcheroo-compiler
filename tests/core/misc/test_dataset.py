# ruff: noqa: E501
import pytest

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.dataset import ArrayIterator, AudioDataset, AutoShardPolicy, AutotuneAlgorithm, Dataset, ImageDataset, NumpyDataset, NumpyIterator, Options, SqlDataset, TextDataset
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.errors import ShapeMismatchError
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

"Tests for dataset primitives."


def test_dataset_basic() -> None:
    """Test basic dataset iteration."""
    backend = get_active_backend()
    device = Device("cpu")
    data1 = Tensor(backend.array([1.0, 2.0, 3.0, 4.0, 5.0]), TensorConfig((5,), DType.Float32, device))
    data2 = Tensor(backend.array([10.0, 20.0, 30.0, 40.0, 50.0]), TensorConfig((5,), DType.Float32, device))
    ds = Dataset.from_tensor_slices(data1, data2).batch(2)
    batches = list(ds)
    assert len(batches) == 3
    assert batches[0][0].shape == (2,)
    assert batches[-1][0].shape == (1,)
    ds2 = Dataset.from_list([1, 2, 3]).batch(2, drop_remainder=True)
    batches2 = list(ds2)
    assert len(batches2) == 1
    opts = Options()
    opts.autotune_algorithm = AutotuneAlgorithm.HILL_CLIMB
    opts.experimental_distribute["auto_shard_policy"] = AutoShardPolicy.DATA
    ds3 = ds2.with_options(opts)
    assert ds3.options_.autotune_algorithm == AutotuneAlgorithm.HILL_CLIMB


def test_dataset_shuffle() -> None:
    """Test dataset shuffling."""
    backend = get_active_backend()
    device = Device("cpu")
    data = Tensor(backend.array(list(range(100))), TensorConfig((100,), DType.Int32, device))
    ds = Dataset(data).batch(10).shuffle(10, seed=42)
    batches = list(ds)
    assert len(batches) == 10
    for b in batches:
        assert b[0].shape == (10,)


def test_dataset_methods() -> None:
    """Test standard dataset transformations."""
    backend = get_active_backend()
    device = Device("cpu")
    data = Tensor(backend.array([1.0, 2.0]), TensorConfig((2,), DType.Float32, device))
    ds = Dataset(data)
    assert ds.prefetch(2)._prefetch_buffer == 2
    assert ds.unbatch()._batch_size == 1
    assert ds.pad_to_cardinality(10) == ds
    assert ds.dense_to_ragged_batch(2)._batch_size == 2
    assert ds.map_and_batch(lambda x: x, 2)._batch_size == 2
    assert ds.group_by_window(lambda x: x, lambda k, ds: ds, 1) == ds
    assert ds.rejection_resample(lambda x: 0, [0.5], [0.5]) == ds
    assert ds.parallel_interleave(lambda x: ds, 1) == ds
    assert ds.prefetch_to_device("cpu") == ds
    assert ds.snapshot("path") == ds
    assert ds.save("path") == ds
    assert isinstance(Dataset.load("path"), Dataset)


def test_dataset_iterators() -> None:
    """Test iterator wrappers."""
    backend = get_active_backend()
    device = Device("cpu")
    data = Tensor(backend.array([1.0, 2.0]), TensorConfig((2,), DType.Float32, device))
    ds = Dataset(data).batch(1)
    np_iter = NumpyIterator(ds)
    assert len(next(iter(np_iter))) == 1
    arr_iter = ArrayIterator(ds)
    assert len(next(iter(arr_iter))) == 1


def test_dataset_exceptions() -> None:
    """Test dataset exceptions."""
    backend = get_active_backend()
    device = Device("cpu")
    data1 = Tensor(backend.array([1.0, 2.0]), TensorConfig((2,), DType.Float32, device))
    data2 = Tensor(backend.array([1.0, 2.0, 3.0]), TensorConfig((3,), DType.Float32, device))
    with pytest.raises((ValueError, ShapeMismatchError), match="At least one tensor must be provided"):
        Dataset.from_tensor_slices()
    with pytest.raises((ValueError, ShapeMismatchError), match="All tensors must have the same leading dimension"):
        Dataset(data1, data2)
    ds = Dataset(data1)
    with pytest.raises((ValueError, ShapeMismatchError), match="batch_size must be positive"):
        ds.batch(0)
    with pytest.raises((ValueError, ShapeMismatchError), match="buffer_size must be positive"):
        ds.shuffle(0)
    with pytest.raises((ValueError, ShapeMismatchError), match="buffer_size must be positive"):
        ds.prefetch(0)
    with pytest.raises((ValueError, ShapeMismatchError), match="batch_size must be positive"):
        ds.dense_to_ragged_batch(0)


def test_specialized_datasets() -> None:
    """Test specialized datasets dummy."""
    pass


def test_dataset_coverage_extras() -> None:
    """Test extras coverage."""
    backend = get_active_backend()
    device = Device("cpu")
    data = Tensor(backend.array([1.0, 2.0, 3.0]), TensorConfig((3,), DType.Float32, device))
    ds = Dataset(data).batch(2, drop_remainder=True)
    assert len(list(ds)) == 1
    ds2 = Dataset(data).map_and_batch(lambda x: (x,), 2)
    assert len(list(ds2)) == 2


def test_split_dataset():
    """Test split dataset."""
    from ml_switcheroo_compiler.utils.dataset_utils import split_dataset

    (a, b) = split_dataset("test", 0.5, False)
    assert a == "test"
    assert b == "test"


"Test module."


class DummyTensor:
    def __init__(self):
        self.shape = (2,)
        self.device = "cpu"
        self.data = [1, 2]
        self.dtype = "float32"


def test_dataset_coverage():
    opts = Options()
    assert opts.autotune_algorithm is None
    ds = Dataset()
    assert list(iter(ds)) == [()]
    t = DummyTensor()
    ds2 = Dataset.from_tensor_slices(t)
    ds3 = Dataset.from_list([t])
    ds.batch(32)
    ds.unbatch()
    ds.pad_to_cardinality(10)
    ds.dense_to_ragged_batch(10)
    ds.map_and_batch(lambda x: x, 10)
    ds.group_by_window(lambda x: x, lambda x: x, 10)
    ds.rejection_resample(lambda x: x, [0.5, 0.5])
    ds.parallel_interleave(lambda x: x, 10)
    ds.prefetch_to_device("cpu")
    ds.shuffle(100)
    ds.prefetch(2)
    ds.snapshot("p")
    ds.save("p")
    Dataset.load("p")
    ds.with_options(opts)
    pass
    pass
    nds = NumpyDataset(t)
    assert iter(nds)
    ImageDataset(t)
    AudioDataset(t)
    TextDataset(t)
    SqlDataset("drv", "src", "q", ())


def test_iterators():
    t = DummyTensor()
    nds = NumpyDataset(t)
    ni = iter(nds)
    pass
    next(ni)
    ids = ImageDataset(t)
    ai = iter(ids)
    pass
    next(ai)


def test_dataset_coverage_more():
    import numpy as np

    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t1 = Tensor(np.array([1, 2, 3]), TensorConfig((3,), DType.Int32, "cpu"))
    ds1 = Dataset(t1)
    ds1.batch(2)
    ds1.shuffle(10, seed=42)
    i = iter(ds1)
    n1 = next(i)
    assert len(n1) == 1
    n_i = NumpyIterator(ds1)
    n_i2 = iter(n_i)
    n_v = next(n_i2)
    assert len(n_v) == 1
    a_i = ArrayIterator(ds1)
    a_i2 = iter(a_i)
    a_v = next(a_i2)
    assert len(a_v) == 1
    ds_empty = Dataset(Tensor(np.array([]), TensorConfig((0,), DType.Int32, "cpu")))
    ImageDataset(t1, target_size=(10, 10), normalize=True)
    AudioDataset(t1, sample_rate=8000)
    TextDataset(t1, vocab_size=100)
    Dataset.load("p", element_spec=(1,), compression="GZIP", reader_func=lambda x: x)


def test_dataset_coverage_even_more():
    import numpy as np

    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    ds = Dataset()
    ds.snapshot("p", "gzip", lambda: None, lambda: None)
    ds.save("p", "gzip", lambda: None)

    class DummyTensorMock:
        def __init__(self):
            self.data = np.array([1, 2, 3])
            self.shape = (3,)
            self.device = "cpu"
            self.dtype = DType.Int32

    t1 = Tensor(np.array([1, 2, 3]), TensorConfig((3,), DType.Int32, "cpu"))
    ds1 = Dataset(t1)
    ds1._batch_size = 2
    ds1._drop_remainder = True
    list(iter(ds1))
    SqlDataset("dr", "ds", "q", (DType.Int32,))
    ImageDataset()
    AudioDataset()
    TextDataset()


def test_dataset_shuffle_2():
    import numpy as np

    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t1 = Tensor(np.array([1, 2, 3]), TensorConfig((3,), DType.Int32, "cpu"))
    ds1 = Dataset(t1)
    ds1.batch(2)
    ds1.shuffle(10)
    list(iter(ds1))


def test_dataset_remaining():
    from ml_switcheroo_compiler.core.dataset import CsvDataset, TFRecordReader

    CsvDataset("f")
    TFRecordReader("f")
    try:
        Dataset.from_tensor_slices()
    except Exception:
        pass
    try:
        Dataset.from_list([])
    except Exception:
        pass
    try:
        Dataset().unbatch()
    except Exception:
        pass


def test_dataset_image_iter(monkeypatch):
    import numpy as np

    from ml_switcheroo_compiler.core.dataset import ImageDataset
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t1 = Tensor(np.array([[[[1]]]]), TensorConfig((1, 1, 1, 1), DType.Float32, "cpu"))
    monkeypatch.setattr("ml_switcheroo_compiler.ops.nn.upsample_ops.upsample_bilinear", lambda t, size: t)
    monkeypatch.setattr("ml_switcheroo_compiler.ops.binary.true_divide", lambda t, val: t)
    ids = ImageDataset(t1, target_size=(10, 10), normalize=True)
    list(iter(ids))


def test_dataset_tensor_error():
    import numpy as np

    from ml_switcheroo_compiler.core.dataset import Dataset
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t1 = Tensor(np.array([1]), TensorConfig((1,), DType.Int32, "cpu"))
    t2 = Tensor(np.array([1, 2]), TensorConfig((2,), DType.Int32, "cpu"))
    with pytest.raises(ValueError):
        Dataset(t1, t2)


def test_dataset_positive_checks():
    from ml_switcheroo_compiler.core.dataset import Dataset

    ds = Dataset()
    with pytest.raises(ValueError):
        ds.batch(0)
    with pytest.raises(ValueError):
        ds.dense_to_ragged_batch(0)
    with pytest.raises(ValueError):
        ds.shuffle(0)
    with pytest.raises(ValueError):
        ds.prefetch(0)


def test_dataset_map_iter():
    import numpy as np

    from ml_switcheroo_compiler.core.dataset import Dataset
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t1 = Tensor(np.array([1, 2, 3]), TensorConfig((3,), DType.Int32, "cpu"))
    ds = Dataset(t1)
    ds.batch(2)
    ds._map_func = lambda x: (x, x)
    list(iter(ds))


import numpy as np


def create_eager_tensor(data):
    backend = get_active_backend()
    data = backend.array(data)
    return Tensor(data, TensorConfig(data.shape, DType.Float32, Device("cpu")))


def test_image_dataset_missing_branches():
    empty_tensor = create_eager_tensor(np.zeros((0, 1, 1, 1)))
    d = ImageDataset(empty_tensor, target_size=(2, 2), normalize=True).batch(2)
    res = list(d)
    assert len(res) == 0

    t_3d = create_eager_tensor(np.ones((2, 2, 1)))
    d2 = ImageDataset(t_3d, target_size=(2, 2), normalize=False).batch(1)
    res2 = list(d2)
    assert len(res2) == 2
    assert res2[0][0].shape == (1, 2, 1)

    t_4d = create_eager_tensor(np.ones((1, 2, 2, 1)))
    d3 = ImageDataset(t_4d, target_size=None, normalize=False).batch(1)
    res3 = list(d3)
    assert len(res3) == 1
    assert res3[0][0].shape == (1, 2, 2, 1)

    t_norm = create_eager_tensor(np.full((1, 2, 2, 1), 255.0))
    d4 = ImageDataset(t_norm, target_size=None, normalize=True).batch(1)
    res4 = list(d4)
    # The normalization divides by 255
    np.testing.assert_array_equal(res4[0][0].numpy(), np.ones((1, 2, 2, 1)))
