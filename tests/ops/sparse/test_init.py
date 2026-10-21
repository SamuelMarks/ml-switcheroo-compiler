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
from tests.testing_utils import as_callable

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
    assert as_callable(as_callable(SparseDenseMatMul)().infer_shape)(t, t) == (2, 3)
    assert as_callable(SparseDenseMatMul)().infer_shape(as_callable(MockSparseTensor)(()), as_callable(MockSparseTensor)(())) == ()
    assert as_callable(as_callable(SparseAdd)().infer_shape)(t, t) == (2, 3)
    assert as_callable(as_callable(SparseSampledAdd)().infer_shape)(t, t) == (2, 3)
    assert as_callable(as_callable(Smm)().infer_shape)(t, t) == (2, 3)
    assert as_callable(Smm)().infer_shape(as_callable(MockSparseTensor)(()), as_callable(MockSparseTensor)(())) == ()
    assert as_callable(as_callable(SparseReduceSum)().infer_shape)(t) == ()
    assert as_callable(as_callable(SparseSoftmax)().infer_shape)(t) == (2, 3)
    assert as_callable(as_callable(SparseReduceMax)().infer_shape)(t) == ()
    assert as_callable(as_callable(SparseBincount)().infer_shape)(t) == (2,)
    assert as_callable(as_callable(SparseCrossHashed)().infer_shape)(t) == (2, 1)
    assert as_callable(as_callable(SparseExpandDims)().infer_shape)(t) == (1, 2, 3)
    assert as_callable(SparseEye)().infer_shape(2) == (2, 2)
    assert as_callable(as_callable(SparseFillEmptyRows)().infer_shape)(t, 1.0) == (2, 3)
    assert as_callable(as_callable(SparseMapValues)().infer_shape)(t) == (2, 3)
    assert as_callable(as_callable(SparseMask)().infer_shape)(t, t) == (2, 3)
    assert as_callable(as_callable(SparseMaximum)().infer_shape)(t, t) == (2, 3)
    assert as_callable(as_callable(SparseMinimum)().infer_shape)(t, t) == (2, 3)
    assert as_callable(as_callable(SparseReorder)().infer_shape)(t) == (2, 3)
    assert as_callable(as_callable(SparseResetShape)().infer_shape)(t) == (2, 3)
    assert as_callable(as_callable(SparseReshape)().infer_shape)(t, (6,)) == (6,)
    assert as_callable(as_callable(SparseRetain)().infer_shape)(t, t) == (2, 3)
    assert as_callable(as_callable(SparseSegmentMean)().infer_shape)(t, t, t) == (1, 3)
    assert as_callable(as_callable(SparseSegmentSqrtN)().infer_shape)(t, t, t) == (1, 3)
    assert as_callable(as_callable(SparseSegmentSum)().infer_shape)(t, t, t) == (1, 3)
    assert as_callable(as_callable(SparseSlice)().infer_shape)(t, 0, 1) == ()
    assert as_callable(as_callable(SparseToIndicator)().infer_shape)(t, 10) == (2, 10)
    assert as_callable(as_callable(SparseTranspose)().infer_shape)(t) == (3, 2)
    assert as_callable(as_callable(RaggedDot)().infer_shape)(t, t) == (2, 3, 3)
    assert as_callable(as_callable(SparseConcat)().infer_shape)(t) == (2, 3)
    assert as_callable(as_callable(SparseSplit)().infer_shape)(t, 2) == ((1, 3), (1, 3))
    assert as_callable(as_callable(SparseToDense)().infer_shape)(t, (2, 3), t, 0.0) == (2, 3)
