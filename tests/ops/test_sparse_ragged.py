"""Test sparse and ragged ops."""

from ml_switcheroo_compiler.ops.ragged import (
    RaggedAdd,
    RaggedDynamicBroadcast,
    RaggedGather,
    RaggedMatMul,
    RaggedTensorToDense,
)
from ml_switcheroo_compiler.ops.sparse import (
    SparseAdd,
    SparseDenseMatMul,
    SparseReduceSum,
    SparseSoftmax,
)


def test_sparse_ops() -> None:
    """Test sparse ops shape inference."""
    op = SparseDenseMatMul()
    assert op.infer_shape(None, None) == ()

    op2 = SparseAdd()

    class Dummy:
        """Class docstring."""

        dense_shape = (2, 2)

    assert op2.infer_shape(Dummy(), None) == (2, 2)

    op3 = SparseReduceSum()
    assert op3.infer_shape(None) == ()

    op4 = SparseSoftmax()
    assert op4.infer_shape(Dummy()) == (2, 2)


def test_ragged_ops() -> None:
    """Test ragged ops shape inference."""
    op = RaggedGather()
    assert op.infer_shape(None, None) == ()

    op2 = RaggedTensorToDense()
    assert op2.infer_shape(None) == ()

    op3 = RaggedAdd()
    assert op3.infer_shape(None, None) == ()

    op4 = RaggedMatMul()
    assert op4.infer_shape(None, None) == ()

    op5 = RaggedDynamicBroadcast()
    assert op5.infer_shape(None, None) == ()
