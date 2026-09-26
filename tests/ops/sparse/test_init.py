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


class MockSparseTensor:
    def __init__(self, shape=()):
        self.shape = shape
        self.dense_shape = shape


def test_sparse_classes_infer_shape():
    t = MockSparseTensor((2, 3))
    assert SparseDenseMatMul().infer_shape(t, t) == (2, 3)
    assert SparseDenseMatMul().infer_shape(MockSparseTensor(()), MockSparseTensor(())) == ()
    assert SparseAdd().infer_shape(t, t) == (2, 3)
    assert SparseSampledAdd().infer_shape(t, t) == (2, 3)
    assert Smm().infer_shape(t, t) == (2, 3)
    assert Smm().infer_shape(MockSparseTensor(()), MockSparseTensor(())) == ()
    assert SparseReduceSum().infer_shape(t) == ()
    assert SparseSoftmax().infer_shape(t) == (2, 3)
    assert SparseReduceMax().infer_shape(t) == ()
    assert SparseBincount().infer_shape(t) == (2,)
    assert SparseCrossHashed().infer_shape(t) == (2, 1)
    assert SparseExpandDims().infer_shape(t) == (1, 2, 3)
    assert SparseEye().infer_shape(2) == (2, 2)
    assert SparseFillEmptyRows().infer_shape(t, 1.0) == (2, 3)
    assert SparseMapValues().infer_shape(t) == (2, 3)
    assert SparseMask().infer_shape(t, t) == (2, 3)
    assert SparseMaximum().infer_shape(t, t) == (2, 3)
    assert SparseMinimum().infer_shape(t, t) == (2, 3)
    assert SparseReorder().infer_shape(t) == (2, 3)
    assert SparseResetShape().infer_shape(t) == (2, 3)
    assert SparseReshape().infer_shape(t, (6,)) == (6,)
    assert SparseRetain().infer_shape(t, t) == (2, 3)
    assert SparseSegmentMean().infer_shape(t, t, t) == (1, 3)
    assert SparseSegmentSqrtN().infer_shape(t, t, t) == (1, 3)
    assert SparseSegmentSum().infer_shape(t, t, t) == (1, 3)
    assert SparseSlice().infer_shape(t, 0, 1) == ()
    assert SparseToIndicator().infer_shape(t, 10) == (2, 10)
    assert SparseTranspose().infer_shape(t) == (3, 2)
    assert RaggedDot().infer_shape(t, t) == (2, 3, 3)
    assert SparseConcat().infer_shape(t) == (2, 3)
    assert SparseSplit().infer_shape(t, 2) == ((1, 3), (1, 3))
    assert SparseToDense().infer_shape(t, (2, 3), t, 0.0) == (2, 3)
