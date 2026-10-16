"""Tests for test_sparse_ragged_coverage."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import sparse_and_ragged as sparse_module

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_sparse_and_ragged_ops() -> None:
    """Test sparse and ragged operations directly and through eager registry.

    Returns:
        None
    """
    x = np.array([1, 2, 2, 3], dtype=np.int32)

    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(cast("Callable[..., EagerTensorProtocol]", sparse_module._np_sparse_bincount)(np, x), cast("Callable[..., EagerTensorProtocol]", np.array)([0, 1, 2, 1]))
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("SparseBincount")(np, x), cast("Callable[..., EagerTensorProtocol]", np.array)([0, 1, 2, 1]))

    assert sparse_module._np_sparse_reduce_max(np, x) == 3
    assert cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("SparseReduceMax")(np, x) == 3

    assert sparse_module._np_sparse_reduce_sum(np, x) == 8
    assert cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("SparseReduceSum")(np, x) == 8

    data = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.int32)
    segment_ids = np.array([0, 1, 1], dtype=np.int32)

    assert sparse_module._np_sparse_segment_mean(np, data, indices, segment_ids) == 20.0
    assert cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("SparseSegmentMean")(np, data, indices, segment_ids) == 20.0

    # SparseSegmentSqrtN non-empty and empty segment_ids
    sqrt_res = sparse_module._np_sparse_segment_sqrt_n(np, data, indices, segment_ids)
    assert cast("Callable[..., EagerTensorProtocol]", np.isclose)(sqrt_res, 60.0 / cast("Callable[..., EagerTensorProtocol]", np.sqrt)(3))
    assert cast("Callable[..., EagerTensorProtocol]", np.isclose)(
        cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("SparseSegmentSqrtN")(np, data, indices, segment_ids),
        60.0 / np.sqrt(3),
    )
    sqrt_res_empty = sparse_module._np_sparse_segment_sqrt_n(np, data, indices, np.array([]))
    assert cast("Callable[..., EagerTensorProtocol]", np.isclose)(sqrt_res_empty, 60.0 / 1.0)

    assert sparse_module._np_sparse_segment_sum(np, data, indices, segment_ids) == 60.0
    assert cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("SparseSegmentSum")(np, data, indices, segment_ids) == 60.0

    # SparseExpandDims branches
    # branch 1: axis=None, kwargs empty
    exp_none = sparse_module._np_sparse_expand_dims(np, data)
    assert cast("EagerTensorProtocol", getattr(exp_none, "shape")) == (3, 1)

    # branch 2: axis=1
    exp_ax = sparse_module._np_sparse_expand_dims(np, data, axis=0)
    assert cast("EagerTensorProtocol", getattr(exp_ax, "shape")) == (1, 3)

    # branch 3: in kwargs
    exp_kw = sparse_module._np_sparse_expand_dims(np, data, **{"axis": 0})
    assert cast("EagerTensorProtocol", getattr(exp_kw, "shape")) == (1, 3)

    reg_exp = numpy_eager_registry.get("SparseExpandDims")
    assert cast("EagerTensorProtocol", getattr(cast("Callable[..., EagerTensorProtocol]", reg_exp)(np, data), "shape")) == (3, 1)

    # RaggedDot
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(cast("Callable[..., EagerTensorProtocol]", sparse_module._np_ragged_dot)(np, a, b), cast("Callable[..., EagerTensorProtocol]", np.dot)(a, b))
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("RaggedDot")(np, a, b), cast("Callable[..., EagerTensorProtocol]", np.dot)(a, b))
