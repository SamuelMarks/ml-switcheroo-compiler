import mlx.core as mx

from ml_switcheroo_compiler.backends.mlx.eager import (
    _execute_numpy_fallback,
    _from_numpy,
    _mlx_cast,
    _mlx_cummax,
    _mlx_cummin,
    _mlx_cumprod,
    _mlx_eye,
    _mlx_full,
    _mlx_nan_to_num,
    _mlx_ones,
    _mlx_partition,
    _mlx_ragged_tensor_to_dense,
    _mlx_reshape,
    _mlx_rope,
    _mlx_scatter_nd,
    _mlx_slice,
    _mlx_take,
    _mlx_take_along_axis,
    _mlx_tensor_scatter_add,
    _mlx_tensor_scatter_max,
    _mlx_tensor_scatter_min,
    _mlx_tensor_scatter_update,
    _mlx_variance,
    _mlx_zeros,
    _parse_partition_k,
    _resolve_dtype,
    _to_numpy,
    execute_op,
)


def test_numpy_conversion_fallback():
    assert _to_numpy([1.0]).shape == (1,)
    assert _from_numpy([1.0]).shape == (1,)
    res = _execute_numpy_fallback(None, "Add", [1.0], [2.0])
    assert res.shape == (1,)

    # execute_op should fall back if op is unknown
    try:
        execute_op(None, "UnknownFakeOp", [1.0])
    except Exception:
        pass

    execute_op(None, "Take", mx.array([1.0]), mx.array([0]), axis=0)
    execute_op(None, "TakeAlongAxis", mx.array([1.0]), mx.array([0]), axis=0)


def test_mlx_ops():
    t = mx.array([1.0])

    class DummyDtype:
        value = "int32"

    _mlx_cast(mx, t, dtype="int32")
    _mlx_cast(mx, t, dtype=None)

    _mlx_ragged_tensor_to_dense(mx, t)
    _mlx_take_along_axis(mx, t, mx.array([0]), axis=0)
    _mlx_take(mx, t, mx.array([0]), axis=0)

    t2 = mx.array([[1.0]])
    _mlx_tensor_scatter_update(mx, t2, mx.array([[0]]), mx.array([2.0]))
    _mlx_tensor_scatter_add(mx, t2, mx.array([[0]]), mx.array([2.0]))
    _mlx_tensor_scatter_max(mx, t2, mx.array([[0]]), mx.array([2.0]))
    _mlx_tensor_scatter_min(mx, t2, mx.array([[0]]), mx.array([2.0]))

    class BoxedShape:
        data = [1]

        def tolist(self):
            return [1]

    _mlx_scatter_nd(mx, mx.array([[0]]), mx.array([1.0]), [1])
    _mlx_scatter_nd(mx, mx.array([[0]]), mx.array([1.0]), shape=[1])
    _mlx_scatter_nd(mx, mx.array([[0]]), mx.array([1.0]), shape=(1,))

    _mlx_reshape(mx, t, shape=BoxedShape())
    _mlx_reshape(mx, t, shape=(1,))

    _resolve_dtype(mx, None)
    _resolve_dtype(mx, "bfloat16")

    _mlx_zeros(mx, BoxedShape(), dtype="int32")
    _mlx_zeros(mx, 1, dtype="int32")
    _mlx_zeros(mx, 1, dtype="fake")

    _mlx_ones(mx, BoxedShape(), dtype="int32")
    _mlx_ones(mx, 1, dtype="int32")
    _mlx_ones(mx, 1, dtype="fake")

    _mlx_full(mx, BoxedShape(), 2.0, dtype="int32")
    _mlx_full(mx, 1, 2.0, dtype="int32")
    _mlx_full(mx, 1, 2.0, dtype="fake")

    class ItemMock:
        def item(self):
            return 1

        def __int__(self):
            return 1

    class DataMock:
        data = ItemMock()

    _parse_partition_k(ItemMock())
    _parse_partition_k(DataMock())

    a = mx.array([[1.0, 2.0, 3.0]])
    _mlx_partition(mx, a, k=1, return_indices=False)
    _mlx_partition(mx, a, k=1, return_indices=True)
    _mlx_partition(mx, a, k=1, return_indices=None)

    _mlx_nan_to_num(mx, t, nan=ItemMock(), posinf=DataMock(), neginf=1.0)

    _mlx_cummax(mx, t, axis=0, dtype="int32")
    _mlx_cummax(mx, t, axis=0, dtype=None)

    _mlx_cummin(mx, t, axis=0, dtype="int32")
    _mlx_cummin(mx, t, axis=0, dtype=None)

    _mlx_cumprod(mx, t, axis=0, dtype="int32")
    _mlx_cumprod(mx, t, axis=0, dtype=None)

    _mlx_slice(mx, t, dim=0, start=0, end=1)

    _mlx_eye(mx, DataMock(), m=DataMock())
    _mlx_eye(mx, 2)

    t3 = mx.array([[[1.0]]])
    _mlx_rope(mx, t3, dim=1)

    _mlx_variance(mx, t)
