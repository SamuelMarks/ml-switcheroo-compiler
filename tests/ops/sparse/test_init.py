"""Tests for test_init."""

from __future__ import annotations

import typing

# ruff: noqa: E501
from ml_switcheroo_compiler.ops.sparse import (
    RaggedDot,
    Smm,
    SparseAdd,
    SparseBincount,
    SparseConcat,
    SparseCrossHashed,
    SparseDenseMatMul,
    SparseExpandDims,
    SparseEye,
    SparseFillEmptyRows,
    SparseMapValues,
    SparseMask,
    SparseMaximum,
    SparseMinimum,
    SparseReduceMax,
    SparseReduceSum,
    SparseReorder,
    SparseResetShape,
    SparseReshape,
    SparseRetain,
    SparseSampledAdd,
    SparseSegmentMean,
    SparseSegmentSqrtN,
    SparseSegmentSum,
    SparseSlice,
    SparseSoftmax,
    SparseSplit,
    SparseToDense,
    SparseToIndicator,
    SparseTranspose,
)
from tests.testing_utils import ensure_callable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


class MockSparseTensor:
    """Test class MockSparseTensor."""

    def __init__(self, shape: EagerValue = ()) -> None:
        """Test function __init__.

        Args:
            shape: Test parameter `shape`.
        """
        self.shape = shape
        self.dense_shape = shape


def test_sparse_classes_infer_shape() -> None:
    """Test function test_sparse_classes_infer_shape."""
    t = MockSparseTensor((2, 3))
    assert ensure_callable(ensure_callable(SparseDenseMatMul)().infer_shape)(t, t) == (2, 3)
    assert ensure_callable(SparseDenseMatMul)().infer_shape(ensure_callable(MockSparseTensor)(()), ensure_callable(MockSparseTensor)(())) == ()
    assert ensure_callable(ensure_callable(SparseAdd)().infer_shape)(t, t) == (2, 3)
    assert ensure_callable(ensure_callable(SparseSampledAdd)().infer_shape)(t, t) == (2, 3)
    assert ensure_callable(ensure_callable(Smm)().infer_shape)(t, t) == (2, 3)
    assert ensure_callable(Smm)().infer_shape(ensure_callable(MockSparseTensor)(()), ensure_callable(MockSparseTensor)(())) == ()
    assert ensure_callable(ensure_callable(SparseReduceSum)().infer_shape)(t) == ()
    assert ensure_callable(ensure_callable(SparseSoftmax)().infer_shape)(t) == (2, 3)
    assert ensure_callable(ensure_callable(SparseReduceMax)().infer_shape)(t) == ()
    assert ensure_callable(ensure_callable(SparseBincount)().infer_shape)(t) == (2,)
    assert ensure_callable(ensure_callable(SparseCrossHashed)().infer_shape)(t) == (2, 1)
    assert ensure_callable(ensure_callable(SparseExpandDims)().infer_shape)(t) == (1, 2, 3)
    assert ensure_callable(SparseEye)().infer_shape(2) == (2, 2)
    assert ensure_callable(ensure_callable(SparseFillEmptyRows)().infer_shape)(t, 1.0) == (2, 3)
    assert ensure_callable(ensure_callable(SparseMapValues)().infer_shape)(t) == (2, 3)
    assert ensure_callable(ensure_callable(SparseMask)().infer_shape)(t, t) == (2, 3)
    assert ensure_callable(ensure_callable(SparseMaximum)().infer_shape)(t, t) == (2, 3)
    assert ensure_callable(ensure_callable(SparseMinimum)().infer_shape)(t, t) == (2, 3)
    assert ensure_callable(ensure_callable(SparseReorder)().infer_shape)(t) == (2, 3)
    assert ensure_callable(ensure_callable(SparseResetShape)().infer_shape)(t) == (2, 3)
    assert ensure_callable(ensure_callable(SparseReshape)().infer_shape)(t, (6,)) == (6,)
    assert ensure_callable(ensure_callable(SparseRetain)().infer_shape)(t, t) == (2, 3)
    assert ensure_callable(ensure_callable(SparseSegmentMean)().infer_shape)(t, t, t) == (1, 3)
    assert ensure_callable(ensure_callable(SparseSegmentSqrtN)().infer_shape)(t, t, t) == (1, 3)
    assert ensure_callable(ensure_callable(SparseSegmentSum)().infer_shape)(t, t, t) == (1, 3)
    assert ensure_callable(ensure_callable(SparseSlice)().infer_shape)(t, 0, 1) == ()
    assert ensure_callable(ensure_callable(SparseToIndicator)().infer_shape)(t, 10) == (2, 10)
    assert ensure_callable(ensure_callable(SparseTranspose)().infer_shape)(t) == (3, 2)
    assert ensure_callable(ensure_callable(RaggedDot)().infer_shape)(t, t) == (2, 3, 3)
    assert ensure_callable(ensure_callable(SparseConcat)().infer_shape)(t) == (2, 3)
    assert ensure_callable(ensure_callable(SparseSplit)().infer_shape)(t, 2) == ((1, 3), (1, 3))
    assert ensure_callable(ensure_callable(SparseToDense)().infer_shape)(t, (2, 3), t, 0.0) == (2, 3)
