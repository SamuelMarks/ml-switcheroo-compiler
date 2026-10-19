"""Tests for test_init."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

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

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerValue


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
    assert cast("EagerCallableProtocol", cast("EagerCallableProtocol", SparseDenseMatMul)().infer_shape)(t, t) == (2, 3)
    assert cast("EagerCallableProtocol", SparseDenseMatMul)().infer_shape(cast("EagerCallableProtocol", MockSparseTensor)(()), cast("EagerCallableProtocol", MockSparseTensor)(())) == ()
    assert cast("EagerCallableProtocol", cast("EagerCallableProtocol", SparseAdd)().infer_shape)(t, t) == (2, 3)
    assert cast("EagerCallableProtocol", cast("EagerCallableProtocol", SparseSampledAdd)().infer_shape)(t, t) == (2, 3)
    assert cast("EagerCallableProtocol", cast("EagerCallableProtocol", Smm)().infer_shape)(t, t) == (2, 3)
    assert cast("EagerCallableProtocol", Smm)().infer_shape(cast("EagerCallableProtocol", MockSparseTensor)(()), cast("EagerCallableProtocol", MockSparseTensor)(())) == ()
    assert cast("EagerCallableProtocol", cast("EagerCallableProtocol", SparseReduceSum)().infer_shape)(t) == ()
    assert cast("EagerCallableProtocol", cast("EagerCallableProtocol", SparseSoftmax)().infer_shape)(t) == (2, 3)
    assert cast("EagerCallableProtocol", cast("EagerCallableProtocol", SparseReduceMax)().infer_shape)(t) == ()
    assert cast("EagerCallableProtocol", cast("EagerCallableProtocol", SparseBincount)().infer_shape)(t) == (2,)
    assert cast("EagerCallableProtocol", cast("EagerCallableProtocol", SparseCrossHashed)().infer_shape)(t) == (2, 1)
    assert cast("EagerCallableProtocol", cast("EagerCallableProtocol", SparseExpandDims)().infer_shape)(t) == (1, 2, 3)
    assert cast("EagerCallableProtocol", SparseEye)().infer_shape(2) == (2, 2)
    assert cast("EagerCallableProtocol", cast("EagerCallableProtocol", SparseFillEmptyRows)().infer_shape)(t, 1.0) == (2, 3)
    assert cast("EagerCallableProtocol", cast("EagerCallableProtocol", SparseMapValues)().infer_shape)(t) == (2, 3)
    assert cast("EagerCallableProtocol", cast("EagerCallableProtocol", SparseMask)().infer_shape)(t, t) == (2, 3)
    assert cast("EagerCallableProtocol", cast("EagerCallableProtocol", SparseMaximum)().infer_shape)(t, t) == (2, 3)
    assert cast("EagerCallableProtocol", cast("EagerCallableProtocol", SparseMinimum)().infer_shape)(t, t) == (2, 3)
    assert cast("EagerCallableProtocol", cast("EagerCallableProtocol", SparseReorder)().infer_shape)(t) == (2, 3)
    assert cast("EagerCallableProtocol", cast("EagerCallableProtocol", SparseResetShape)().infer_shape)(t) == (2, 3)
    assert cast("EagerCallableProtocol", cast("EagerCallableProtocol", SparseReshape)().infer_shape)(t, (6,)) == (6,)
    assert cast("EagerCallableProtocol", cast("EagerCallableProtocol", SparseRetain)().infer_shape)(t, t) == (2, 3)
    assert cast("EagerCallableProtocol", cast("EagerCallableProtocol", SparseSegmentMean)().infer_shape)(t, t, t) == (1, 3)
    assert cast("EagerCallableProtocol", cast("EagerCallableProtocol", SparseSegmentSqrtN)().infer_shape)(t, t, t) == (1, 3)
    assert cast("EagerCallableProtocol", cast("EagerCallableProtocol", SparseSegmentSum)().infer_shape)(t, t, t) == (1, 3)
    assert cast("EagerCallableProtocol", cast("EagerCallableProtocol", SparseSlice)().infer_shape)(t, 0, 1) == ()
    assert cast("EagerCallableProtocol", cast("EagerCallableProtocol", SparseToIndicator)().infer_shape)(t, 10) == (2, 10)
    assert cast("EagerCallableProtocol", cast("EagerCallableProtocol", SparseTranspose)().infer_shape)(t) == (3, 2)
    assert cast("EagerCallableProtocol", cast("EagerCallableProtocol", RaggedDot)().infer_shape)(t, t) == (2, 3, 3)
    assert cast("EagerCallableProtocol", cast("EagerCallableProtocol", SparseConcat)().infer_shape)(t) == (2, 3)
    assert cast("EagerCallableProtocol", cast("EagerCallableProtocol", SparseSplit)().infer_shape)(t, 2) == ((1, 3), (1, 3))
    assert cast("EagerCallableProtocol", cast("EagerCallableProtocol", SparseToDense)().infer_shape)(t, (2, 3), t, 0.0) == (2, 3)
