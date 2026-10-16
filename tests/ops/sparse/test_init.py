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
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


class MockSparseTensor:
    """Test class MockSparseTensor."""

    def __init__(self, shape: EagerValue = ()) -> None:
        """Test function __init__."""
        self.shape = shape
        self.dense_shape = shape


def test_sparse_classes_infer_shape() -> None:
    """Test function test_sparse_classes_infer_shape."""
    t = MockSparseTensor((2, 3))
    assert cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", SparseDenseMatMul)().infer_shape)(t, t) == (2, 3)
    assert cast("Callable[..., EagerTensorProtocol]", SparseDenseMatMul)().infer_shape(cast("Callable[..., EagerTensorProtocol]", MockSparseTensor)(()), cast("Callable[..., EagerTensorProtocol]", MockSparseTensor)(())) == ()
    assert cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", SparseAdd)().infer_shape)(t, t) == (2, 3)
    assert cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", SparseSampledAdd)().infer_shape)(t, t) == (2, 3)
    assert cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", Smm)().infer_shape)(t, t) == (2, 3)
    assert cast("Callable[..., EagerTensorProtocol]", Smm)().infer_shape(cast("Callable[..., EagerTensorProtocol]", MockSparseTensor)(()), cast("Callable[..., EagerTensorProtocol]", MockSparseTensor)(())) == ()
    assert cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", SparseReduceSum)().infer_shape)(t) == ()
    assert cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", SparseSoftmax)().infer_shape)(t) == (2, 3)
    assert cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", SparseReduceMax)().infer_shape)(t) == ()
    assert cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", SparseBincount)().infer_shape)(t) == (2,)
    assert cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", SparseCrossHashed)().infer_shape)(t) == (2, 1)
    assert cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", SparseExpandDims)().infer_shape)(t) == (1, 2, 3)
    assert cast("Callable[..., EagerTensorProtocol]", SparseEye)().infer_shape(2) == (2, 2)
    assert cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", SparseFillEmptyRows)().infer_shape)(t, 1.0) == (2, 3)
    assert cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", SparseMapValues)().infer_shape)(t) == (2, 3)
    assert cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", SparseMask)().infer_shape)(t, t) == (2, 3)
    assert cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", SparseMaximum)().infer_shape)(t, t) == (2, 3)
    assert cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", SparseMinimum)().infer_shape)(t, t) == (2, 3)
    assert cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", SparseReorder)().infer_shape)(t) == (2, 3)
    assert cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", SparseResetShape)().infer_shape)(t) == (2, 3)
    assert cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", SparseReshape)().infer_shape)(t, (6,)) == (6,)
    assert cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", SparseRetain)().infer_shape)(t, t) == (2, 3)
    assert cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", SparseSegmentMean)().infer_shape)(t, t, t) == (1, 3)
    assert cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", SparseSegmentSqrtN)().infer_shape)(t, t, t) == (1, 3)
    assert cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", SparseSegmentSum)().infer_shape)(t, t, t) == (1, 3)
    assert cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", SparseSlice)().infer_shape)(t, 0, 1) == ()
    assert cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", SparseToIndicator)().infer_shape)(t, 10) == (2, 10)
    assert cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", SparseTranspose)().infer_shape)(t) == (3, 2)
    assert cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", RaggedDot)().infer_shape)(t, t) == (2, 3, 3)
    assert cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", SparseConcat)().infer_shape)(t) == (2, 3)
    assert cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", SparseSplit)().infer_shape)(t, 2) == ((1, 3), (1, 3))
    assert cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", SparseToDense)().infer_shape)(t, (2, 3), t, 0.0) == (2, 3)
