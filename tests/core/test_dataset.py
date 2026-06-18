"""Tests for dataset primitives."""

import pytest

from ml_switcheroo_compiler.core.dataset import Dataset
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.device import Device


def test_dataset_basic():
    """Test basic dataset iteration."""
    backend = get_active_backend()
    device = Device("cpu")

    data1 = Tensor(backend.array([1.0, 2.0, 3.0, 4.0, 5.0]), (5,), DType.Float32, device)
    data2 = Tensor(backend.array([10.0, 20.0, 30.0, 40.0, 50.0]), (5,), DType.Float32, device)

    ds = Dataset(data1, data2).batch(2)
    batches = list(ds)

    assert len(batches) == 3
    assert batches[0][0].shape == (2,)
    assert batches[-1][0].shape == (1,)


def test_dataset_shuffle():
    """Test dataset shuffling."""
    backend = get_active_backend()
    device = Device("cpu")

    data = Tensor(backend.array(list(range(100))), (100,), DType.Int32, device)

    ds = Dataset(data).batch(10).shuffle(10)
    batches = list(ds)

    # Just check it runs and produces right shapes
    assert len(batches) == 10
    for b in batches:
        assert b[0].shape == (10,)


def test_dataset_prefetch():
    """Test dataset prefetching."""
    backend = get_active_backend()
    device = Device("cpu")

    data = Tensor(backend.array([1.0, 2.0]), (2,), DType.Float32, device)
    ds = Dataset(data).prefetch(2)
    assert ds._prefetch_buffer == 2


def test_dataset_exceptions():
    """Test dataset exceptions."""
    backend = get_active_backend()
    device = Device("cpu")

    data1 = Tensor(backend.array([1.0, 2.0]), (2,), DType.Float32, device)
    data2 = Tensor(backend.array([1.0, 2.0, 3.0]), (3,), DType.Float32, device)

    with pytest.raises(ValueError, match="At least one tensor must be provided"):
        Dataset()

    with pytest.raises(ValueError, match="All tensors must have the same leading dimension"):
        Dataset(data1, data2)

    ds = Dataset(data1)
    with pytest.raises(ValueError, match="batch_size must be positive"):
        ds.batch(0)

    with pytest.raises(ValueError, match="buffer_size must be positive"):
        ds.shuffle(0)

    with pytest.raises(ValueError, match="buffer_size must be positive"):
        ds.prefetch(0)
