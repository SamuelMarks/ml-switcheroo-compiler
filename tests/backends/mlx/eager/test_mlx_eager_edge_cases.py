import mlx.core as mx
import numpy as np
import pytest

import ml_switcheroo_compiler.backends.mlx.eager as mlx_eager


def test_mlx_eager_coverage():
    # _to_numpy / _from_numpy
    res = mlx_eager._to_numpy(1.0)
    assert isinstance(res, np.ndarray)
    res = mlx_eager._from_numpy(res)
    assert isinstance(res, mx.array)

    # execute_op
    res = mlx_eager.execute_op(None, "Zeros", shape=(2,))
    assert res is not None

    with pytest.raises(Exception):
        mlx_eager.execute_op(None, "UnknownOp")

    res = mlx_eager.execute_op(None, "TakeAlongAxis", mx.array([1, 2]), mx.array([0]), axis=0)
    assert res is not None

    res = mlx_eager.execute_op(None, "Take", mx.array([1, 2]), mx.array([0]), axis=0)
    assert res is not None

    # _mlx_cast
    res = mlx_eager._mlx_cast(mx, mx.array([1.0]), dtype=mx.int32)
    assert res is not None
    res = mlx_eager._mlx_cast(mx, mx.array([1.0]), mx.int32)
    assert res is not None
    res = mlx_eager._mlx_cast(mx, mx.array([1.0]), dtype=None)
    assert res is not None

    class DummyDtype:
        def __str__(self):
            return "int32"

        value = "int32"

    res = mlx_eager._mlx_cast(mx, mx.array([1.0]), dtype=DummyDtype())
    assert res is not None

    # _mlx_ragged_tensor_to_dense
    res = mlx_eager._mlx_ragged_tensor_to_dense(mx, 1)
    assert res == 1

    # _mlx_take_along_axis, _mlx_take
    res = mlx_eager._mlx_take_along_axis(mx, mx.array([1, 2]), mx.array([0]), axis=0)
    assert res is not None
    res = mlx_eager._mlx_take(mx, mx.array([1, 2]), mx.array([0]), axis=0)
    assert res is not None

    # _mlx_tensor_scatter_update
    res = mlx_eager._mlx_tensor_scatter_update(mx, mx.array([1, 2]), mx.array([[0]]), mx.array([3]))
    assert res is not None

    # _mlx_tensor_scatter_add, _mlx_tensor_scatter_max, _mlx_tensor_scatter_min
    res = mlx_eager._mlx_tensor_scatter_add(mx, mx.array([1]), mx.array([0]), mx.array([1]))
    assert res is not None
    res = mlx_eager._mlx_tensor_scatter_max(mx, mx.array([1]), mx.array([0]), mx.array([1]))
    assert res is not None
    res = mlx_eager._mlx_tensor_scatter_min(mx, mx.array([1]), mx.array([0]), mx.array([1]))
    assert res is not None

    # _mlx_scatter_nd
    res = mlx_eager._mlx_scatter_nd(mx, mx.array([0]), mx.array([1]), shape=mx.array([1]))
    assert res is not None
    res = mlx_eager._mlx_scatter_nd(mx, mx.array([0]), mx.array([1]), (1,))
    assert res is not None

    # _mlx_reshape
    res = mlx_eager._mlx_reshape(mx, mx.array([1]), shape=mx.array([1]))
    assert res is not None
    res = mlx_eager._mlx_reshape(mx, mx.array([1]), shape=np.array([1]))
    assert res is not None
    res = mlx_eager._mlx_reshape(mx, mx.array([1]), mx.array([1]))
    assert res is not None
    res = mlx_eager._mlx_reshape(mx, input=mx.array([1]), shape=(1,))
    assert res is not None

    # _resolve_dtype
    res = mlx_eager._resolve_dtype(mx, None)
    assert res is None
    res = mlx_eager._resolve_dtype(mx, "float32")
    assert res is not None
    res = mlx_eager._resolve_dtype(mx, "bfloat16")
    assert res is not None

    # _mlx_zeros, _mlx_ones, _mlx_full
    res = mlx_eager._mlx_zeros(mx, mx.array([1]))
    assert res is not None
    res = mlx_eager._mlx_zeros(mx, np.array([1]))
    assert res is not None
    res = mlx_eager._mlx_zeros(mx, 1)
    assert res is not None
    res = mlx_eager._mlx_zeros(mx, shape=(1,), dtype=None)
    assert res is not None

    res = mlx_eager._mlx_ones(mx, mx.array([1]))
    assert res is not None
    res = mlx_eager._mlx_ones(mx, np.array([1]))
    assert res is not None
    res = mlx_eager._mlx_ones(mx, 1)
    assert res is not None
    res = mlx_eager._mlx_ones(mx, shape=(1,), dtype=None)
    assert res is not None

    res = mlx_eager._mlx_full(mx, mx.array([1]), 1.0)
    assert res is not None
    res = mlx_eager._mlx_full(mx, np.array([1]), 1.0)
    assert res is not None
    res = mlx_eager._mlx_full(mx, 1.0, 1.0)
    assert res is not None
    res = mlx_eager._mlx_full(mx, shape=(1,), fill_value=1.0, dtype=None)
    assert res is not None

    # _parse_partition_k
    assert mlx_eager._parse_partition_k(mx.array([1])) == 1

    class ItemMock:
        def item(self):
            return 1

    assert mlx_eager._parse_partition_k(ItemMock()) == 1

    class DataItemMock:
        data = ItemMock()

    assert mlx_eager._parse_partition_k(DataItemMock()) == 1
    assert mlx_eager._parse_partition_k(1) == 1

    # _mlx_partition
    res = mlx_eager._mlx_partition(mx, mx.array([2, 1, 3]), 1)
    assert res is not None
    res = mlx_eager._mlx_partition(mx, mx.array([2, 1, 3]), k=1, return_indices=False)
    assert res is not None
    res = mlx_eager._mlx_partition(mx, mx.array([2, 1, 3]), k=1, return_indices=True)
    assert res is not None

    # _mlx_nan_to_num
    res = mlx_eager._mlx_nan_to_num(mx, mx.array([float("nan"), 1.0]), nan=0.0)
    assert res is not None
    res = mlx_eager._mlx_nan_to_num(mx, mx.array([float("nan"), 1.0]), nan=ItemMock())
    assert res is not None
    res = mlx_eager._mlx_nan_to_num(mx, mx.array([float("nan"), 1.0]), nan=DataItemMock())
    assert res is not None
    res = mlx_eager._mlx_nan_to_num(mx, mx.array([float("nan"), 1.0]), posinf=None)
    assert res is not None

    # _mlx_cummax, _mlx_cummin, _mlx_cumprod
    res = mlx_eager._mlx_cummax(mx, mx.array([1, 2]), axis=0, dtype=DummyDtype())
    assert res is not None
    res = mlx_eager._mlx_cummin(mx, mx.array([1, 2]), axis=0, dtype=DummyDtype())
    assert res is not None
    res = mlx_eager._mlx_cumprod(mx, mx.array([1, 2]), axis=0, dtype=DummyDtype())
    assert res is not None
    res = mlx_eager._mlx_cummax(mx, mx.array([1, 2]), axis=0, dtype=None)
    assert res is not None
    res = mlx_eager._mlx_cummin(mx, mx.array([1, 2]), axis=0, dtype=None)
    assert res is not None
    res = mlx_eager._mlx_cumprod(mx, mx.array([1, 2]), axis=0, dtype=None)
    assert res is not None

    # _mlx_slice
    res = mlx_eager._mlx_slice(mx, mx.array([1, 2]), dim=0, start=0, end=1)
    assert res is not None

    # _mlx_eye
    res = mlx_eager._mlx_eye(mx, mx.array([2]))
    assert res is not None
    res = mlx_eager._mlx_eye(mx, mx.array([1]), mx.array([1]))
    assert res is not None
    res = mlx_eager._mlx_eye(mx, mx.array([2]), mx.array([2]))
    assert res is not None

    # _mlx_rope
    res = mlx_eager._mlx_rope(mx, mx.array([[[[1.0, 2.0]]]]), dim=2)
    assert res is not None

    # _mlx_variance
    res = mlx_eager._mlx_variance(mx, mx.array([1.0, 2.0]))
    assert res is not None
