"""Unit tests for sparse operations shape inference, batch dimensions, and broadcasting."""

from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.sparse import (
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
    """Mock sparse tensor containing dense_shape metadata."""

    def __init__(self, dense_shape: tuple[int, ...]) -> None:
        """Initialize mock sparse tensor.

        Args:
            dense_shape (tuple[int, ...]): Equivalent dense shape.
        """
        self.dense_shape = dense_shape
        self.shape = dense_shape


def _make_tensor(shape: tuple[int, ...]) -> Tensor:
    """Helper to create dummy tensor with specified shape.

    Args:
        shape (tuple[int, ...]): Shape of tensor.

    Returns:
        Tensor: Dummy tensor instance.
    """
    return Tensor(None, TensorConfig(shape, "float32", "cpu"))


def test_sparse_dense_matmul_and_smm() -> None:
    """Verify SparseDenseMatMul and Smm shape inference with batch broadcasting."""
    op_spmm = SparseDenseMatMul()
    op_smm = Smm()

    # 2D: (M=4, K=8) x (K=8, N=16) -> (4, 16)
    sp_a = MockSparseTensor((4, 8))
    dense_b = _make_tensor((8, 16))
    assert op_spmm.infer_shape(sp_a, dense_b) == (4, 16)
    assert op_smm.infer_shape(sp_a, dense_b) == (4, 16)

    # Batched broadcasting: (2, 1, 5, 10) x (1, 3, 10, 6) -> (2, 3, 5, 6)
    sp_batch = MockSparseTensor((2, 1, 5, 10))
    dense_batch = _make_tensor((1, 3, 10, 6))
    assert op_spmm.infer_shape(sp_batch, dense_batch) == (2, 3, 5, 6)
    assert op_smm.infer_shape(sp_batch, dense_batch) == (2, 3, 5, 6)

    # Fallbacks
    assert op_spmm.infer_shape(None, None) == ()
    assert op_smm.infer_shape(None, None) == ()


def test_sparse_elementwise_broadcasting() -> None:
    """Verify SparseAdd, SparseSampledAdd, SparseMaximum, SparseMinimum broadcasting."""
    op_add = SparseAdd()
    op_s_add = SparseSampledAdd()
    op_max = SparseMaximum()
    op_min = SparseMinimum()

    # Broadcast (2, 1, 4) and (1, 3, 4) -> (2, 3, 4)
    sp1 = MockSparseTensor((2, 1, 4))
    sp2 = MockSparseTensor((1, 3, 4))

    assert op_add.infer_shape(sp1, sp2) == (2, 3, 4)
    assert op_s_add.infer_shape(sp1, sp2) == (2, 3, 4)
    assert op_max.infer_shape(sp1, sp2) == (2, 3, 4)
    assert op_min.infer_shape(sp1, sp2) == (2, 3, 4)


def test_sparse_reductions() -> None:
    """Verify SparseReduceSum and SparseReduceMax shape reductions."""
    op_sum = SparseReduceSum()
    op_max = SparseReduceMax()

    sp = MockSparseTensor((4, 8, 16))

    # Reduce along axis 1 (without keepdims)
    assert op_sum.infer_shape(sp, axis=1, keepdims=False) == (4, 16)
    assert op_max.infer_shape(sp, axis=1, keepdims=False) == (4, 16)

    # Reduce along axis 1 (with keepdims)
    assert op_sum.infer_shape(sp, axis=1, keepdims=True) == (4, 1, 16)
    assert op_max.infer_shape(sp, axis=1, keepdims=True) == (4, 1, 16)

    # Full reduction
    assert op_sum.infer_shape(sp, axis=None, keepdims=False) == ()
    assert op_max.infer_shape(sp, axis=None, keepdims=False) == ()


def test_sparse_shape_preserving_ops() -> None:
    """Verify shape preserving sparse operations."""
    op_softmax = SparseSoftmax()
    op_map = SparseMapValues()
    op_mask = SparseMask()
    op_fill = SparseFillEmptyRows()
    op_reorder = SparseReorder()
    op_retain = SparseRetain()

    sp = MockSparseTensor((5, 10, 15))

    assert op_softmax.infer_shape(sp) == (5, 10, 15)
    assert op_map.infer_shape(sp) == (5, 10, 15)
    assert op_mask.infer_shape(sp) == (5, 10, 15)
    assert op_fill.infer_shape(sp) == (5, 10, 15)
    assert op_reorder.infer_shape(sp) == (5, 10, 15)
    assert op_retain.infer_shape(sp) == (5, 10, 15)


def test_sparse_reshaping_and_dimensions() -> None:
    """Verify SparseExpandDims, SparseReshape, SparseResetShape, SparseTranspose."""
    op_expand = SparseExpandDims()
    op_reshape = SparseReshape()
    op_reset = SparseResetShape()
    op_transpose = SparseTranspose()

    sp = MockSparseTensor((4, 8))

    # Expand dims at axis 1 -> (4, 1, 8)
    assert op_expand.infer_shape(sp, axis=1) == (4, 1, 8)

    # Reshape to (2, 2, 8)
    assert op_reshape.infer_shape(sp, (2, 2, 8)) == (2, 2, 8)

    # Reset shape to (8, 4)
    assert op_reset.infer_shape(sp, new_shape=(8, 4)) == (8, 4)

    # Transpose (4, 8) -> (8, 4)
    assert op_transpose.infer_shape(sp) == (8, 4)

    # Transpose with perm
    sp_3d = MockSparseTensor((2, 4, 8))
    assert op_transpose.infer_shape(sp_3d, perm=(2, 0, 1)) == (8, 2, 4)


def test_sparse_concat_and_split() -> None:
    """Verify SparseConcat and SparseSplit shape inference."""
    op_concat = SparseConcat()
    op_split = SparseSplit()

    sp1 = MockSparseTensor((4, 6))
    sp2 = MockSparseTensor((4, 10))

    # Concat along axis 1 -> (4, 16)
    assert op_concat.infer_shape([sp1, sp2], axis=1) == (4, 16)

    # Split (4, 16) into 4 chunks along axis 1 -> 4 tuples of (4, 4)
    sp_split_in = MockSparseTensor((4, 16))
    chunks = op_split.infer_shape(sp_split_in, num_split=4, axis=1)
    assert len(chunks) == 4
    for ch in chunks:
        assert ch == (4, 4)


def test_sparse_generation_and_conversions() -> None:
    """Verify SparseEye, SparseBincount, SparseCrossHashed, SparseToIndicator, SparseToDense."""
    op_eye = SparseEye()
    op_bincount = SparseBincount()
    op_cross = SparseCrossHashed()
    op_to_ind = SparseToIndicator()
    op_to_dense = SparseToDense()
    op_slice = SparseSlice()

    # SparseEye
    assert op_eye.infer_shape(5, num_columns=7) == (5, 7)
    assert op_eye.infer_shape(4) == (4, 4)

    # SparseBincount
    sp = MockSparseTensor((10,))
    assert op_bincount.infer_shape(sp, size=50) == (10, 50)

    # SparseCrossHashed
    assert op_cross.infer_shape([sp], num_buckets=100) == (10, 100)

    # SparseToIndicator
    assert op_to_ind.infer_shape(sp, vocab_size=256) == (10, 256)

    # SparseToDense
    assert op_to_dense.infer_shape(None, (10, 20)) == (10, 20)

    # SparseSlice
    sp_2d = MockSparseTensor((10, 20))
    assert op_slice.infer_shape(sp_2d, start=(2, 4), size=(3, 5)) == (3, 5)


def test_sparse_segment_operations() -> None:
    """Verify SparseSegmentSum, SparseSegmentMean, SparseSegmentSqrtN."""
    op_sum = SparseSegmentSum()
    op_mean = SparseSegmentMean()
    op_sqrt = SparseSegmentSqrtN()

    data = _make_tensor((20, 8))

    assert op_sum.infer_shape(data, num_segments=5) == (5, 8)
    assert op_mean.infer_shape(data, num_segments=5) == (5, 8)
    assert op_sqrt.infer_shape(data, num_segments=5) == (5, 8)


def test_sparse_remaining_branches_and_edge_cases() -> None:
    """Test sparse edge cases covering all remaining lines and branch decisions."""
    from ml_switcheroo_compiler.ops.sparse import (
        SparseConcat,
        SparseExpandDims,
        SparseReduceMax,
        SparseReshape,
        SparseSlice,
        SparseSplit,
        _broadcast_sparse_shapes,
        _get_dense_shape,
    )

    # 1. _get_dense_shape with non-container scalar and list/tuple (line 56)
    assert _get_dense_shape(12345) == ()
    assert _get_dense_shape([10, 20]) == (10, 20)

    # 2. _broadcast_sparse_shapes when shape_a is empty (line 96)
    assert _broadcast_sparse_shapes((), (2, 3)) == (2, 3)

    # 3. SparseReduceMax with empty shape (line 263)
    op_red_max = SparseReduceMax()
    assert op_red_max.infer_shape(None) == ()

    # 4. SparseExpandDims with negative axis (line 333)
    op_expand = SparseExpandDims()
    assert op_expand.infer_shape(MockSparseTensor((2, 3)), axis=-1) == (2, 3, 1)

    # 5. SparseReshape with shape=None (line 518)
    op_reshape = SparseReshape()
    assert op_reshape.infer_shape(MockSparseTensor((10, 20)), shape=None) == (10, 20)

    # 6. SparseSlice with size=None (line 631)
    op_slice = SparseSlice()
    assert op_slice.infer_shape(MockSparseTensor((10, 20)), start=(0, 0), size=None) == (10, 20)

    # 7. SparseConcat: empty inputs, negative axis, and short shapes (lines 722, 726, 730->728, 732->734)
    op_concat = SparseConcat()
    assert op_concat.infer_shape([]) == ()
    sp_a = MockSparseTensor((4, 8))
    sp_b = MockSparseTensor((4, 8))
    assert op_concat.infer_shape([sp_a, sp_b], axis=-1) == (4, 16)
    # len(s) <= axis and len(first_shape) <= axis
    sp_1d = MockSparseTensor((4,))
    sp_empty = MockSparseTensor(())
    assert op_concat.infer_shape([sp_1d, sp_empty], axis=2) == (4,)

    # 8. SparseSplit: negative axis and axis >= len(shape) (lines 759, 760->762)
    op_split = SparseSplit()
    assert op_split.infer_shape(MockSparseTensor((4, 8)), num_split=2, axis=-1) == ((4, 4), (4, 4))
    assert op_split.infer_shape(MockSparseTensor((4,)), num_split=2, axis=2) == ((4,), (4,))
