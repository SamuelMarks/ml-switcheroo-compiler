"""Tests for test_sparse_ragged_coverage."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import sparse_and_ragged as sparse_module


def test_sparse_and_ragged_ops() -> None:
    """Test sparse and ragged operations directly and through eager registry.

    Returns:
        None
    """
    x = np.array([1, 2, 2, 3], dtype=np.int32)

    assert np.array_equal(sparse_module._np_sparse_bincount(np, x), np.array([0, 1, 2, 1]))
    assert np.array_equal(numpy_eager_registry.get("SparseBincount")(np, x), np.array([0, 1, 2, 1]))

    assert sparse_module._np_sparse_reduce_max(np, x) == 3
    assert numpy_eager_registry.get("SparseReduceMax")(np, x) == 3

    assert sparse_module._np_sparse_reduce_sum(np, x) == 8
    assert numpy_eager_registry.get("SparseReduceSum")(np, x) == 8

    data = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.int32)
    segment_ids = np.array([0, 1, 1], dtype=np.int32)

    assert sparse_module._np_sparse_segment_mean(np, data, indices, segment_ids) == 20.0
    assert numpy_eager_registry.get("SparseSegmentMean")(np, data, indices, segment_ids) == 20.0

    # SparseSegmentSqrtN non-empty and empty segment_ids
    sqrt_res = sparse_module._np_sparse_segment_sqrt_n(np, data, indices, segment_ids)
    assert np.isclose(sqrt_res, 60.0 / np.sqrt(3))
    assert np.isclose(
        numpy_eager_registry.get("SparseSegmentSqrtN")(np, data, indices, segment_ids),
        60.0 / np.sqrt(3),
    )
    sqrt_res_empty = sparse_module._np_sparse_segment_sqrt_n(np, data, indices, np.array([]))
    assert np.isclose(sqrt_res_empty, 60.0 / 1.0)

    assert sparse_module._np_sparse_segment_sum(np, data, indices, segment_ids) == 60.0
    assert numpy_eager_registry.get("SparseSegmentSum")(np, data, indices, segment_ids) == 60.0

    # SparseExpandDims branches
    # branch 1: axis=None, kwargs empty
    exp_none = sparse_module._np_sparse_expand_dims(np, data)
    assert exp_none.shape == (3, 1)

    # branch 2: axis=1
    exp_ax = sparse_module._np_sparse_expand_dims(np, data, axis=0)
    assert exp_ax.shape == (1, 3)

    # branch 3: in kwargs
    exp_kw = sparse_module._np_sparse_expand_dims(np, data, **{"axis": 0})
    assert exp_kw.shape == (1, 3)

    reg_exp = numpy_eager_registry.get("SparseExpandDims")
    assert reg_exp(np, data).shape == (3, 1)

    # RaggedDot
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    assert np.array_equal(sparse_module._np_ragged_dot(np, a, b), np.dot(a, b))
    assert np.array_equal(numpy_eager_registry.get("RaggedDot")(np, a, b), np.dot(a, b))
