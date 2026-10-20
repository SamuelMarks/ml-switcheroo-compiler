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

if typing.TYPE_CHECKING:
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
    assert typing.cast("EagerCallableProtocol", typing.cast("EagerCallableProtocol", SparseDenseMatMul)().infer_shape)(t, t) == (2, 3)
    assert typing.cast("EagerCallableProtocol", SparseDenseMatMul)().infer_shape(typing.cast("EagerCallableProtocol", MockSparseTensor)(()), typing.cast("EagerCallableProtocol", MockSparseTensor)(())) == ()
    assert typing.cast("EagerCallableProtocol", typing.cast("EagerCallableProtocol", SparseAdd)().infer_shape)(t, t) == (2, 3)
    assert typing.cast("EagerCallableProtocol", typing.cast("EagerCallableProtocol", SparseSampledAdd)().infer_shape)(t, t) == (2, 3)
    assert typing.cast("EagerCallableProtocol", typing.cast("EagerCallableProtocol", Smm)().infer_shape)(t, t) == (2, 3)
    assert typing.cast("EagerCallableProtocol", Smm)().infer_shape(typing.cast("EagerCallableProtocol", MockSparseTensor)(()), typing.cast("EagerCallableProtocol", MockSparseTensor)(())) == ()
    assert typing.cast("EagerCallableProtocol", typing.cast("EagerCallableProtocol", SparseReduceSum)().infer_shape)(t) == ()
    assert typing.cast("EagerCallableProtocol", typing.cast("EagerCallableProtocol", SparseSoftmax)().infer_shape)(t) == (2, 3)
    assert typing.cast("EagerCallableProtocol", typing.cast("EagerCallableProtocol", SparseReduceMax)().infer_shape)(t) == ()
    assert typing.cast("EagerCallableProtocol", typing.cast("EagerCallableProtocol", SparseBincount)().infer_shape)(t) == (2,)
    assert typing.cast("EagerCallableProtocol", typing.cast("EagerCallableProtocol", SparseCrossHashed)().infer_shape)(t) == (2, 1)
    assert typing.cast("EagerCallableProtocol", typing.cast("EagerCallableProtocol", SparseExpandDims)().infer_shape)(t) == (1, 2, 3)
    assert typing.cast("EagerCallableProtocol", SparseEye)().infer_shape(2) == (2, 2)
    assert typing.cast("EagerCallableProtocol", typing.cast("EagerCallableProtocol", SparseFillEmptyRows)().infer_shape)(t, 1.0) == (2, 3)
    assert typing.cast("EagerCallableProtocol", typing.cast("EagerCallableProtocol", SparseMapValues)().infer_shape)(t) == (2, 3)
    assert typing.cast("EagerCallableProtocol", typing.cast("EagerCallableProtocol", SparseMask)().infer_shape)(t, t) == (2, 3)
    assert typing.cast("EagerCallableProtocol", typing.cast("EagerCallableProtocol", SparseMaximum)().infer_shape)(t, t) == (2, 3)
    assert typing.cast("EagerCallableProtocol", typing.cast("EagerCallableProtocol", SparseMinimum)().infer_shape)(t, t) == (2, 3)
    assert typing.cast("EagerCallableProtocol", typing.cast("EagerCallableProtocol", SparseReorder)().infer_shape)(t) == (2, 3)
    assert typing.cast("EagerCallableProtocol", typing.cast("EagerCallableProtocol", SparseResetShape)().infer_shape)(t) == (2, 3)
    assert typing.cast("EagerCallableProtocol", typing.cast("EagerCallableProtocol", SparseReshape)().infer_shape)(t, (6,)) == (6,)
    assert typing.cast("EagerCallableProtocol", typing.cast("EagerCallableProtocol", SparseRetain)().infer_shape)(t, t) == (2, 3)
    assert typing.cast("EagerCallableProtocol", typing.cast("EagerCallableProtocol", SparseSegmentMean)().infer_shape)(t, t, t) == (1, 3)
    assert typing.cast("EagerCallableProtocol", typing.cast("EagerCallableProtocol", SparseSegmentSqrtN)().infer_shape)(t, t, t) == (1, 3)
    assert typing.cast("EagerCallableProtocol", typing.cast("EagerCallableProtocol", SparseSegmentSum)().infer_shape)(t, t, t) == (1, 3)
    assert typing.cast("EagerCallableProtocol", typing.cast("EagerCallableProtocol", SparseSlice)().infer_shape)(t, 0, 1) == ()
    assert typing.cast("EagerCallableProtocol", typing.cast("EagerCallableProtocol", SparseToIndicator)().infer_shape)(t, 10) == (2, 10)
    assert typing.cast("EagerCallableProtocol", typing.cast("EagerCallableProtocol", SparseTranspose)().infer_shape)(t) == (3, 2)
    assert typing.cast("EagerCallableProtocol", typing.cast("EagerCallableProtocol", RaggedDot)().infer_shape)(t, t) == (2, 3, 3)
    assert typing.cast("EagerCallableProtocol", typing.cast("EagerCallableProtocol", SparseConcat)().infer_shape)(t) == (2, 3)
    assert typing.cast("EagerCallableProtocol", typing.cast("EagerCallableProtocol", SparseSplit)().infer_shape)(t, 2) == ((1, 3), (1, 3))
    assert typing.cast("EagerCallableProtocol", typing.cast("EagerCallableProtocol", SparseToDense)().infer_shape)(t, (2, 3), t, 0.0) == (2, 3)
