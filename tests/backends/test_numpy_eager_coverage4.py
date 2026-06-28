import numpy as np

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.ops.aliases.array_ops import ediff1d
from ml_switcheroo_compiler.backends.numpy.eager.indexing import (
    _np_dynamic_slice_in_dim,
    _np_dynamic_update_slice_in_dim,
    _np_dynamic_index_in_dim,
    _np_dynamic_update_index_in_dim,
    _np_slice_in_dim,
    _np_scatter_apply,
    _np_scatter_max,
    _np_scatter_min,
    _np_scatter_mul,
)
from ml_switcheroo_compiler.backends.numpy.eager.math import (
    _np_clz,
    _np_bitcast_convert_type,
    _np_sort_key_val,
    _np_reduce_precision,
)
from ml_switcheroo_compiler.backends.numpy.eager.reductions import _np_cumlogsumexp


def test_ediff1d_coverage():
    with ConfigContext(eager_mode=True):
        res1 = ediff1d([1, 2, 4], to_end=[5], to_begin=[0])
        assert res1 is not None


def test_numpy_eager_indexing_extra_coverage():
    operand = np.arange(10)

    # DynamicSliceInDim
    res = _np_dynamic_slice_in_dim(None, operand, start_index=2, slice_size=3, axis=0)
    np.testing.assert_array_equal(res, [2, 3, 4])

    # DynamicUpdateSliceInDim
    res = _np_dynamic_update_slice_in_dim(
        None, operand, update=np.array([9, 9, 9]), start_index=2, axis=0
    )
    expected = np.arange(10)
    expected[2:5] = [9, 9, 9]
    np.testing.assert_array_equal(res, expected)

    # DynamicIndexInDim
    res = _np_dynamic_index_in_dim(None, operand, index=5, axis=0, keepdims=True)
    np.testing.assert_array_equal(res, [5])
    res = _np_dynamic_index_in_dim(None, operand, index=5, axis=0, keepdims=False)
    assert res == 5

    # DynamicUpdateIndexInDim
    res = _np_dynamic_update_index_in_dim(None, operand, update=99, index=5, axis=0)
    expected = np.arange(10)
    expected[5] = 99
    np.testing.assert_array_equal(res, expected)

    # SliceInDim
    res = _np_slice_in_dim(None, operand, start_index=1, limit_index=6, stride=2, axis=0)
    np.testing.assert_array_equal(res, [1, 3, 5])

    # ScatterApply
    res = _np_scatter_apply(None, operand, None, None, None)
    np.testing.assert_array_equal(res, operand)

    # ScatterMax, Min, Mul
    operand2 = np.zeros(5)
    indices = np.array([[1], [3]])
    updates = np.array([5, 10])

    res = _np_scatter_max(None, operand2, indices, updates)
    expected = np.zeros(5)
    expected[1] = 5
    expected[3] = 10
    np.testing.assert_array_equal(res, expected)

    operand2 = np.ones(5) * 20
    res = _np_scatter_min(None, operand2, indices, updates)
    expected = np.ones(5) * 20
    expected[1] = 5
    expected[3] = 10
    np.testing.assert_array_equal(res, expected)

    operand2 = np.ones(5) * 2
    res = _np_scatter_mul(None, operand2, indices, updates)
    expected = np.ones(5) * 2
    expected[1] = 10
    expected[3] = 20
    np.testing.assert_array_equal(res, expected)


def test_numpy_eager_math_extra_coverage():
    # _np_clz coverage
    res64 = _np_clz(None, np.array([1, 2], dtype=np.int64))
    assert res64.shape == (2,)
    res8 = _np_clz(None, np.array([1, 2], dtype=np.uint8))
    assert res8.shape == (2,)
    res16 = _np_clz(None, np.array([1, 2], dtype=np.int16))
    assert res16.shape == (2,)

    # _np_bitcast_convert_type
    res_bitcast = _np_bitcast_convert_type(None, np.array([1.0], dtype=np.float32), "int32")
    assert res_bitcast.dtype == np.int32

    # _np_sort_key_val
    keys = np.array([3, 1, 2])
    vals = np.array([30, 10, 20])
    s_keys, s_vals = _np_sort_key_val(None, keys, vals, axis=0)
    np.testing.assert_array_equal(s_keys, [1, 2, 3])
    np.testing.assert_array_equal(s_vals, [10, 20, 30])


def test_numpy_eager_reductions_extra():
    res = _np_cumlogsumexp(None, np.array([1, 2, 3]), axis=0)
    assert res is not None


def test_numpy_eager_math_extra_coverage2():
    # _np_clz for int32
    res = _np_clz(None, np.array([1, 2], dtype=np.int32))
    assert res.shape == (2,)

    # _np_reduce_precision
    res = _np_reduce_precision(None, np.array([1.123]), 5, 10)
    assert res is not None
