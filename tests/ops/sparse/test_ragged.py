"""Test sparse and ragged ops."""

from __future__ import annotations

import numpy as np
import pytest

from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.ragged import (
    RaggedAdd,
    RaggedDynamicBroadcast,
    RaggedGather,
    RaggedMatMul,
    RaggedTensorToDense,
)
from ml_switcheroo_compiler.ops.ragged.core import RaggedDot
from ml_switcheroo_compiler.ops.sparse import (
    Smm,
    SparseAdd,
    SparseDenseMatMul,
    SparseReduceSum,
    SparseSampledAdd,
    SparseSoftmax,
)
from tests.ops.linalg.decompositions.test_decompositions import _test_op
from tests.testing_utils import adapt_backend_module, extract_test_shape, unpack_pair


def test_sparse_ops() -> None:
    """Test the sparse ops behavior.

    Returns:
        OpValue: The inferred shape or computed result.
    """
    op = SparseDenseMatMul()
    assert op.infer_shape(None, None) == ()
    op2 = SparseAdd()

    class Dummy:
        """Configuration class for dummy."""

        dense_shape = (2, 2)
        shape = (2, 2)

    assert op2.infer_shape(Dummy(), None) == (2, 2)
    op3 = SparseReduceSum()
    assert op3.infer_shape(None) == ()
    op4 = SparseSoftmax()
    assert op4.infer_shape(Dummy()) == (2, 2)
    assert op.infer_shape(Dummy(), Dummy()) == (2, 2)
    op5 = SparseSampledAdd()
    assert op5.infer_shape(Dummy(), None) == (2, 2)
    op6 = Smm()
    assert op6.infer_shape(Dummy(), Dummy()) == (2, 2)


def test_ragged_ops() -> None:
    """Test the ragged ops behavior.

    Returns:
        OpValue: The inferred shape or computed result.
    """
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


a = Tensor(np.array([1.0, 2.0]), TensorConfig((2,), "float32", "cpu"))


def test_sparse_bincount() -> None:
    """Test the sparse bincount behavior."""
    with pytest.raises(AttributeError):
        import ml_switcheroo_compiler.ops as _ops_mod

        sparse_bincount = adapt_backend_module(_ops_mod).require_symbol("sparse_bincount")
        (e, t) = unpack_pair(_test_op(sparse_bincount, a))
        assert e is not None


def test_sparse_reduce_max() -> None:
    """Test the sparse reduce max behavior."""
    with pytest.raises(AttributeError):
        import ml_switcheroo_compiler.ops as _ops_mod

        sparse_reduce_max = adapt_backend_module(_ops_mod).require_symbol("sparse_reduce_max")
        (e, t) = unpack_pair(_test_op(sparse_reduce_max, a))
        assert e is not None


def test_sparse_reduce_sum() -> None:
    """Test the sparse reduce sum behavior."""
    with pytest.raises(AttributeError):
        import ml_switcheroo_compiler.ops as _ops_mod

        sparse_reduce_sum = adapt_backend_module(_ops_mod).require_symbol("sparse_reduce_sum")
        (e, t) = unpack_pair(_test_op(sparse_reduce_sum, a))
        assert e is not None


def test_sparse_segment_mean() -> None:
    """Test the sparse segment mean behavior."""
    from ml_switcheroo_compiler.ops import sparse_segment_mean

    (e, t) = unpack_pair(_test_op(sparse_segment_mean, a, a, a))
    assert tuple(extract_test_shape(e)) == ()


def test_sparse_segment_sqrt_n() -> None:
    """Test the sparse segment sqrt n behavior."""
    with pytest.raises(AttributeError):
        import ml_switcheroo_compiler.ops as _ops_mod

        sparse_segment_sqrt_n = adapt_backend_module(_ops_mod).require_symbol("sparse_segment_sqrt_n")
        (e, t) = unpack_pair(_test_op(sparse_segment_sqrt_n, a, a, a))
        assert e is not None


def test_sparse_segment_sum() -> None:
    """Test the sparse segment sum behavior."""
    with pytest.raises(AttributeError):
        import ml_switcheroo_compiler.ops as _ops_mod

        sparse_segment_sum = adapt_backend_module(_ops_mod).require_symbol("sparse_segment_sum")
        (e, t) = unpack_pair(_test_op(sparse_segment_sum, a, a, a))
        assert e is not None


def test_ragged_dot() -> None:
    """Test the ragged dot behavior."""
    with pytest.raises(AttributeError):
        import ml_switcheroo_compiler.ops as _ops_mod

        ragged_dot = adapt_backend_module(_ops_mod).require_symbol("ragged_dot")
        (e, t) = unpack_pair(_test_op(ragged_dot, a, a))
        assert e is not None


def test_ragged_missing() -> None:
    """Test function test_ragged_missing."""
    assert RaggedDot().infer_shape() == ()

    class DummyShape:
        """Test class DummyShape."""

        shape = (1, 2)

    assert RaggedDot().infer_shape(DummyShape()) == (1, 2, 1)
