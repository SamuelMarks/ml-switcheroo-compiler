"""Behavioral tests for NumPy eager sparse and ragged operations."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.backends.eager import sparse_ops as sparse_module
from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from tests.testing_utils import coerce_float, coerce_int, extract_test_shape, require_callable


def test_sparse_bincount_and_reductions() -> None:
    """Verify `SparseBincount`, `SparseReduceMax`, and `SparseReduceSum` compute exact counts and reductions."""
    x = np.array([1, 2, 2, 3], dtype=np.int32)
    expected_counts = np.array([0, 1, 2, 1])

    np.testing.assert_array_equal(ensure_ndarray(sparse_module._np_sparse_bincount(np, x)), expected_counts)
    reg_bincount = numpy_eager_registry.get("SparseBincount")
    np.testing.assert_array_equal(ensure_ndarray(require_callable(reg_bincount)(np, x)), expected_counts)

    assert coerce_int(sparse_module._np_sparse_reduce_max(np, x)) == 3
    reg_max = numpy_eager_registry.get("SparseReduceMax")
    assert require_callable(coerce_int)(require_callable(reg_max)(np, x)) == 3

    assert coerce_int(sparse_module._np_sparse_reduce_sum(np, x)) == 8
    reg_sum = numpy_eager_registry.get("SparseReduceSum")
    assert require_callable(coerce_int)(require_callable(reg_sum)(np, x)) == 8


def test_sparse_segment_reductions() -> None:
    """Verify `SparseSegmentMean`, `SparseSegmentSqrtN`, and `SparseSegmentSum` across non-empty and empty segments."""
    data = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.int32)
    segment_ids = np.array([0, 1, 1], dtype=np.int32)

    assert coerce_float(sparse_module._np_sparse_segment_mean(np, data, indices, segment_ids)) == 20.0
    reg_mean = numpy_eager_registry.get("SparseSegmentMean")
    assert require_callable(coerce_float)(require_callable(reg_mean)(np, data, indices, segment_ids)) == 20.0

    sqrt_res = coerce_float(sparse_module._np_sparse_segment_sqrt_n(np, data, indices, segment_ids))
    np.testing.assert_allclose(sqrt_res, 60.0 / np.sqrt(3.0))
    reg_sqrt = numpy_eager_registry.get("SparseSegmentSqrtN")
    np.testing.assert_allclose(
        require_callable(coerce_float)(require_callable(reg_sqrt)(np, data, indices, segment_ids)), 60.0 / np.sqrt(3.0)
    )

    sqrt_empty = coerce_float(sparse_module._np_sparse_segment_sqrt_n(np, data, indices, np.array([])))
    np.testing.assert_allclose(sqrt_empty, 60.0)

    assert coerce_float(sparse_module._np_sparse_segment_sum(np, data, indices, segment_ids)) == 60.0
    reg_seg_sum = numpy_eager_registry.get("SparseSegmentSum")
    assert require_callable(coerce_float)(require_callable(reg_seg_sum)(np, data, indices, segment_ids)) == 60.0


def test_sparse_expand_dims_and_ragged_dot() -> None:
    """Verify `SparseExpandDims` axis handling and `_np_ragged_dot` matrix multiplication."""
    data = np.array([10.0, 20.0, 30.0], dtype=np.float32)

    exp_default = ensure_ndarray(sparse_module._np_sparse_expand_dims(np, data))
    assert exp_default.shape == (3, 1)

    exp_pos_axis = ensure_ndarray(sparse_module._np_sparse_expand_dims(np, data, axis=0))
    assert exp_pos_axis.shape == (1, 3)

    exp_kw_axis = ensure_ndarray(sparse_module._np_sparse_expand_dims(np, data, **{"axis": 0}))
    assert exp_kw_axis.shape == (1, 3)

    reg_exp = numpy_eager_registry.get("SparseExpandDims")
    assert extract_test_shape(require_callable(reg_exp)(np, data)) == (3, 1)

    lhs = np.array([[1, 2], [3, 4]])
    rhs = np.array([[5, 6], [7, 8]])
    expected_dot = np.array([[19, 22], [43, 50]])
    np.testing.assert_array_equal(ensure_ndarray(sparse_module._np_ragged_dot(np, lhs, rhs)), expected_dot)
