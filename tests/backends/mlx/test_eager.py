import sys
from unittest.mock import MagicMock, patch

import pytest

# Force mlx.core to be a mock so we have a consistent test environment
mock_mx = MagicMock()

# Inject into sys.modules BEFORE importing eager.py
sys.modules["mlx"] = MagicMock()
sys.modules["mlx.core"] = mock_mx
sys.modules["mlx.core.distributed"] = MagicMock()

import ml_switcheroo_compiler.backends.mlx.eager as eager

# also override any local reference eager.mx
eager.mx = mock_mx


def test_get_mlx_func():
    assert eager._get_mlx_func("Mul") == mock_mx.multiply
    assert eager._get_mlx_func("Sub") == mock_mx.subtract
    assert eager._get_mlx_func("Div") == mock_mx.divide

    # mock getattr to simulate missing attributes for NotFound
    del mock_mx.not_found
    del mock_mx.linalg.not_found
    del mock_mx.fft.not_found
    assert eager._get_mlx_func("NotFound") is None


def test_execute_op():
    mock_mx.zeros = MagicMock(return_value="zeros")
    assert eager.execute_op(None, "Zeros", shape=(1,)) == "zeros"

    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

    global_eager_registry.register("TestOp")(lambda m, *args, **kwargs: "global")
    assert eager.execute_op(None, "TestOp") == "global"

    mock_mx.add = MagicMock(return_value="add")
    assert eager.execute_op(None, "Add") == "add"

    from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

    with patch("ml_switcheroo_compiler.backends.mlx.eager._get_mlx_func", return_value=None):
        with pytest.raises(BackendNotSupportedError):
            eager.execute_op(None, "NotFoundOpXYZ")


def test_mlx_cast():
    t = MagicMock()
    t.astype.return_value = "casted"
    assert eager._mlx_cast(mock_mx, t, dtype="float32") == "casted"
    assert eager._mlx_cast(mock_mx, t, "float32") == "casted"
    assert eager._mlx_cast(mock_mx, t, dtype=None) == t


def test_ragged_tensor_to_dense():
    assert eager._mlx_ragged_tensor_to_dense(mock_mx, "tensor") == "tensor"


def test_take_along_axis():
    mock_mx.take_along_axis = MagicMock(return_value="take_res")
    assert eager._mlx_take_along_axis(mock_mx, "a", "i") == "take_res"


def test_take():
    mock_mx.take = MagicMock(return_value="take_res")
    assert eager._mlx_take(mock_mx, "a", "i") == "take_res"


def test_tensor_scatter_update():
    res = MagicMock()
    mock_mx.array = MagicMock(return_value=res)
    indices = MagicMock()
    indices.shape = (2, 2, 2)
    updates = MagicMock()
    eager._mlx_tensor_scatter_update(mock_mx, "tensor", indices, updates)
    res.__setitem__.assert_called()


def test_tensor_scatter_add():
    res = MagicMock()
    res.__getitem__.return_value = 1
    mock_mx.array = MagicMock(return_value=res)
    indices = MagicMock()
    indices.shape = (2, 2)
    eager._mlx_tensor_scatter_add(mock_mx, "tensor", indices, 2)
    res.__setitem__.assert_called_with(tuple([indices[..., 0], indices[..., 1]]), 3)


def test_tensor_scatter_max():
    res = MagicMock()
    mock_mx.array.return_value = res
    mock_mx.maximum = MagicMock(return_value="max")
    indices = MagicMock()
    indices.shape = (1,)
    with patch("mlx.core.maximum", return_value="max"):
        eager._mlx_tensor_scatter_max(mock_mx, "tensor", indices, 5)
    res.__setitem__.assert_called_with(tuple([indices[..., 0]]), mock_mx.maximum())


def test_tensor_scatter_min():
    res = MagicMock()
    mock_mx.array.return_value = res
    mock_mx.minimum = MagicMock(return_value="min")
    indices = MagicMock()
    indices.shape = (1,)
    with patch("mlx.core.minimum", return_value="min"):
        eager._mlx_tensor_scatter_min(mock_mx, "tensor", indices, 5)
    res.__setitem__.assert_called_with(tuple([indices[..., 0]]), mock_mx.minimum())


def test_scatter_nd():
    mock_mx.zeros = MagicMock()
    res = MagicMock()
    mock_mx.zeros.return_value = res
    updates = MagicMock()
    indices = MagicMock()
    indices.shape = (1,)
    eager._mlx_scatter_nd(mock_mx, indices, updates, (2, 2))
    eager._mlx_scatter_nd(mock_mx, indices, updates, shape=(2, 2))
    shape_mock = MagicMock()
    shape_mock.data.tolist.return_value = [2, 2]
    eager._mlx_scatter_nd(mock_mx, indices, updates, shape=shape_mock)


def test_reshape():
    mock_mx.reshape = MagicMock(return_value="reshape")
    eager._mlx_reshape(mock_mx, "input", (2, 2))
    eager._mlx_reshape(mock_mx, "input", shape=(2, 2))
    eager._mlx_reshape(mock_mx, input="input", newshape=(2, 2))
    shape_mock = MagicMock()
    shape_mock.data.tolist.return_value = [2, 2]
    eager._mlx_reshape(mock_mx, "input", shape=shape_mock)


def test_resolve_dtype():
    mock_mx.float32 = "real_float32"

    assert eager._resolve_dtype(mock_mx, None) is None
    assert eager._resolve_dtype(mock_mx, "float32") == "real_float32"
    # test fallback
    del mock_mx.bfloat16
    mock_mx.float32 = "real_float32"
    assert eager._resolve_dtype(mock_mx, "bfloat16") == "real_float32"
    del mock_mx.unsupported
    assert eager._resolve_dtype(mock_mx, "unsupported") == "unsupported"


def test_zeros():
    mock_mx.zeros = MagicMock(return_value="zeros")
    eager._mlx_zeros(mock_mx, (1,), dtype=None)
    with patch("ml_switcheroo_compiler.backends.mlx.eager._resolve_dtype", return_value="float32"):
        eager._mlx_zeros(mock_mx, (1,), dtype="float32")
    with patch("ml_switcheroo_compiler.backends.mlx.eager._resolve_dtype", return_value="float32"):
        # We need it to raise TypeError then return "zeros2"
        mock_mx.zeros.side_effect = [TypeError(), "zeros2"]
        eager._mlx_zeros(mock_mx, (1,), dtype="float32")
    shape_mock = MagicMock()
    shape_mock.data = (1,)
    # reset side effect
    mock_mx.zeros.side_effect = None
    eager._mlx_zeros(mock_mx, shape=shape_mock)
    eager._mlx_zeros(mock_mx, 5.0)
    eager._mlx_zeros(mock_mx, [1, 2])

    # cover line 312: hasattr(shape, "data") where shape is not 1D (already handled basically but maybe shape has .data but isinstance doesn't change it to float)
    class TupleShape:
        data = (1, 2)

    eager._mlx_zeros(mock_mx, TupleShape())


def test_ones():
    mock_mx.ones = MagicMock(return_value="ones")
    eager._mlx_ones(mock_mx, (1,), dtype=None)
    with patch("ml_switcheroo_compiler.backends.mlx.eager._resolve_dtype", return_value="float32"):
        eager._mlx_ones(mock_mx, (1,), dtype="float32")
    with patch("ml_switcheroo_compiler.backends.mlx.eager._resolve_dtype", return_value="float32"):
        mock_mx.ones.side_effect = [TypeError(), "ones2"]
        eager._mlx_ones(mock_mx, (1,), dtype="float32")
    shape_mock = MagicMock()
    shape_mock.data = (1,)
    mock_mx.ones.side_effect = None
    eager._mlx_ones(mock_mx, shape=shape_mock)
    eager._mlx_ones(mock_mx, 5)


def test_full():
    mock_mx.full = MagicMock(return_value="full")
    eager._mlx_full(mock_mx, (1,), 5, dtype=None)
    with patch("ml_switcheroo_compiler.backends.mlx.eager._resolve_dtype", return_value="float32"):
        eager._mlx_full(mock_mx, (1,), 5, dtype="float32")
    with patch("ml_switcheroo_compiler.backends.mlx.eager._resolve_dtype", return_value="float32"):
        mock_mx.full.side_effect = [TypeError(), "full2"]
        eager._mlx_full(mock_mx, (1,), 5, dtype="float32")
    shape_mock = MagicMock()
    shape_mock.data = (1,)
    mock_mx.full.side_effect = None
    eager._mlx_full(mock_mx, shape=shape_mock, fill_value=5)
    eager._mlx_full(mock_mx, 5, 5)


def test_parse_partition_k():
    assert eager._parse_partition_k(5) == 5
    m = MagicMock()
    m.item.return_value = 5
    assert eager._parse_partition_k(m) == 5
    m2 = MagicMock()
    del m2.item
    m2.data.item.return_value = 5
    assert eager._parse_partition_k(m2) == 5


def test_partition():
    mock_mx.topk = MagicMock(return_value="topk")
    mock_mx.argpartition = MagicMock()
    mock_mx.argpartition.return_value = MagicMock()
    mock_mx.take_along_axis = MagicMock(return_value="values")
    a = MagicMock()
    a.shape = (10,)
    assert eager._mlx_partition(mock_mx, a, 2, return_indices=False) == "topk"

    del mock_mx.topk
    mock_mx.partition = MagicMock(return_value=MagicMock())
    eager._mlx_partition(mock_mx, a, 2, return_indices=False)
    eager._mlx_partition(mock_mx, a, 2, return_indices=True)
    eager._mlx_partition(mock_mx, a, 2, return_indices=None)


def test_nan_to_num():
    mock_mx.nan_to_num = MagicMock(return_value="res")
    val = MagicMock()
    val.item.return_value = 1.0
    val2 = MagicMock()
    del val2.item
    val2.data.item.return_value = 2.0
    assert eager._mlx_nan_to_num(mock_mx, "tensor", nan=val, posinf=val2, neginf=3.0) == "res"
    assert eager._mlx_nan_to_num(mock_mx, "tensor", posinf=None) == "res"


def test_cum_ops():
    mock_mx.cummax = MagicMock()
    mock_mx.cummax().astype.return_value = "res"
    mock_mx.cummin = MagicMock()
    mock_mx.cummin().astype.return_value = "res"
    mock_mx.cumprod = MagicMock()
    mock_mx.cumprod().astype.return_value = "res"
    dtype = MagicMock()
    dtype.value = "float32"
    assert eager._mlx_cummax(mock_mx, "tensor", dtype=dtype) == "res"
    assert eager._mlx_cummin(mock_mx, "tensor", dtype="float32") == "res"
    assert eager._mlx_cummax(mock_mx, "tensor", dtype="float32") == "res"
    assert eager._mlx_cumprod(mock_mx, "tensor", dtype="float32") == "res"
    # branches where dtype loop continues without match
    assert eager._mlx_cummax(mock_mx, "tensor", dtype="float32") == "res"
    assert eager._mlx_cumprod(mock_mx, "tensor", dtype=dtype) == "res"

    mock_mx.cummax = MagicMock(return_value="res")
    assert eager._mlx_cummax(mock_mx, "tensor", dtype=None) == "res"
    mock_mx.cummin = MagicMock(return_value="res")
    assert eager._mlx_cummin(mock_mx, "tensor", dtype=None) == "res"
    mock_mx.cumprod = MagicMock(return_value="res")
    assert eager._mlx_cumprod(mock_mx, "tensor", dtype=None) == "res"


def test_slice():
    a = MagicMock()
    a.shape = (2, 2)
    eager._mlx_slice(mock_mx, a, dim=1, start=0, end=1)
    a.__getitem__.assert_called()


def test_eye():
    mock_mx.eye = MagicMock(return_value="eye")
    mock_mx.float32 = "float32"
    n = MagicMock()
    n.data = 2
    m = MagicMock()
    m.data = 3
    assert eager._mlx_eye(mock_mx, n, m, k=1) == "eye"
    assert eager._mlx_eye(mock_mx, 2) == "eye"


def test_rope():
    mock_mx.fast = MagicMock()
    mock_mx.fast.rope.return_value = "rope"
    assert eager._mlx_rope(mock_mx, "x", dim=2) == "rope"


def test_variance():
    mock_mx.var = MagicMock(return_value="var")
    assert eager._mlx_variance(mock_mx, "tensor") == "var"


def test_distributed_ops():
    mock_mx.distributed = MagicMock()
    mock_mx.distributed.all_sum.return_value = "all_sum"
    mock_mx.distributed.all_gather.return_value = "all_gather"
    mock_mx.distributed.all_to_all.return_value = "all_to_all"
    mock_mx.distributed.recv.return_value = "recv"
    with patch("mlx.core.distributed.all_sum", return_value="all_sum", create=True):
        assert eager._mlx_all_reduce(mock_mx, "tensor") == "all_sum"
    with patch("mlx.core.distributed.all_gather", return_value="all_gather", create=True):
        assert eager._mlx_all_gather(mock_mx, "tensor") == "all_gather"
    with patch("mlx.core.distributed.all_to_all", return_value="all_to_all", create=True):
        assert eager._mlx_all_to_all(mock_mx, "tensor") == "all_to_all"

    mock_reduced = MagicMock()
    mock_reduced.shape = (4, 4)
    mock_reduced.ndim = 2
    mock_reduced.__getitem__.return_value = "sliced_tensor"
    mock_mx.distributed.all_sum.return_value = mock_reduced
    with patch("mlx.core.distributed.all_sum", return_value=mock_reduced, create=True) as mock_all_sum:
        assert eager._mlx_reduce_scatter(mock_mx, "tensor") == "sliced_tensor"

    mock_mx.distributed = None
    assert eager._mlx_all_reduce(mock_mx, "tensor") == "tensor"

    mock_mx.expand_dims = MagicMock(return_value="expand")
    assert eager._mlx_all_gather(mock_mx, "tensor") == "expand"
    assert eager._mlx_all_to_all(mock_mx, "tensor") == "tensor"

    from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

    with pytest.raises(BackendNotSupportedError):
        eager._mlx_reduce_scatter(mock_mx, "tensor")


def test_mlx_execute_op_attribute_error_raise():
    from ml_switcheroo_compiler.backends.mlx.eager import execute_op
    from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

    def mock_get_func(op):
        raise AttributeError("mock missing")

    with patch("ml_switcheroo_compiler.backends.mlx.eager._get_mlx_func", side_effect=mock_get_func):
        with patch("ml_switcheroo_compiler.backends.eager_registry.global_eager_registry.get", return_value=None):
            with patch("ml_switcheroo_compiler.backends.eager_registry.mlx_eager_registry.get", return_value=None):
                with pytest.raises(BackendNotSupportedError):
                    execute_op(None, "OpThatThrowsAttr", 5)


def test_mlx_execute_op_attribute_error_none():
    from ml_switcheroo_compiler.backends.mlx.eager import execute_op
    from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

    with patch("ml_switcheroo_compiler.backends.mlx.eager._get_mlx_func", return_value=None):
        with patch("ml_switcheroo_compiler.backends.eager_registry.global_eager_registry.get", return_value=None):
            with patch("ml_switcheroo_compiler.backends.eager_registry.mlx_eager_registry.get", return_value=None):
                with pytest.raises(BackendNotSupportedError):
                    execute_op(None, "OpWithoutFunc", 5)


def test_mlx_cum_ops_dtype_fallbacks():
    import sys

    from ml_switcheroo_compiler.backends.mlx.eager import _mlx_cummax, _mlx_cummin, _mlx_cumprod

    mx = sys.modules["mlx.core"]
    mx.cummax = MagicMock()
    mx.cummax().astype.return_value = "res"
    mx.cummin = MagicMock()
    mx.cummin().astype.return_value = "res"
    mx.cumprod = MagicMock()
    mx.cumprod().astype.return_value = "res"

    class MockDtype:
        value = "float16"

        def __str__(self):
            return "float16"

    with patch("ml_switcheroo_compiler.backends.mlx.eager._resolve_dtype", return_value="real_dtype"):
        _mlx_cummax(mx, "t", dtype=MockDtype())
        _mlx_cummin(mx, "t", dtype="float16")
        _mlx_cumprod(mx, "t", dtype=MockDtype())
