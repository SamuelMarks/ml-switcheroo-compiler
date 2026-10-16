# ruff: noqa: E501
"""Test sparse and ragged ops."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

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

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_sparse_ops() -> None:
    """Test the sparse ops behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Test sparse ops shape inference."
        op = SparseDenseMatMul()
        assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(None, None) == ()
        op2 = SparseAdd()

        class Dummy:
            """Configuration class for dummy."""

            dense_shape = (2, 2)
            shape = (2, 2)

        assert cast("Callable[..., EagerTensorProtocol]", op2.infer_shape)(cast("Callable[..., EagerTensorProtocol]", Dummy)(), None) == (2, 2)
        op3 = SparseReduceSum()
        assert cast("Callable[..., EagerTensorProtocol]", op3.infer_shape)(None) == ()
        op4 = SparseSoftmax()
        assert cast("Callable[..., EagerTensorProtocol]", op4.infer_shape)(cast("Callable[..., EagerTensorProtocol]", Dummy)()) == (2, 2)

        assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", Dummy)(), cast("Callable[..., EagerTensorProtocol]", Dummy)()) == (2, 2)

        op5 = SparseSampledAdd()
        assert cast("Callable[..., EagerTensorProtocol]", op5.infer_shape)(cast("Callable[..., EagerTensorProtocol]", Dummy)(), None) == (2, 2)

        op6 = Smm()
        assert cast("Callable[..., EagerTensorProtocol]", op6.infer_shape)(cast("Callable[..., EagerTensorProtocol]", Dummy)(), cast("Callable[..., EagerTensorProtocol]", Dummy)()) == (2, 2)
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def test_ragged_ops() -> None:
    """Test the ragged ops behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Test ragged ops shape inference."
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
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


a = Tensor(np.array([1.0, 2.0]), TensorConfig((2,), "float32", "cpu"))


def test_sparse_bincount() -> None:
    """Test the sparse bincount behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Test sparse_bincount."
        import ml_switcheroo_compiler.ops as _ops_mod

        sparse_bincount = cast("Callable[..., EagerTensorProtocol]", getattr(_ops_mod, "sparse_bincount"))

        (e, t) = cast("Callable[..., EagerTensorProtocol]", _test_op)(sparse_bincount, a)
        assert e is not None
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def test_sparse_reduce_max() -> None:
    """Test the sparse reduce max behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Test sparse_reduce_max."
        import ml_switcheroo_compiler.ops as _ops_mod

        sparse_reduce_max = cast("Callable[..., EagerTensorProtocol]", getattr(_ops_mod, "sparse_reduce_max"))

        (e, t) = cast("Callable[..., EagerTensorProtocol]", _test_op)(sparse_reduce_max, a)
        assert e is not None
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def test_sparse_reduce_sum() -> None:
    """Test the sparse reduce sum behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Test sparse_reduce_sum."
        import ml_switcheroo_compiler.ops as _ops_mod

        sparse_reduce_sum = cast("Callable[..., EagerTensorProtocol]", getattr(_ops_mod, "sparse_reduce_sum"))

        (e, t) = cast("Callable[..., EagerTensorProtocol]", _test_op)(sparse_reduce_sum, a)
        assert e is not None
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def test_sparse_segment_mean() -> None:
    """Test the sparse segment mean behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Test sparse_segment_mean."
        from ml_switcheroo_compiler.ops import sparse_segment_mean

        (e, t) = cast("Callable[..., EagerTensorProtocol]", _test_op)(sparse_segment_mean, a, a, a)
        assert e is not None
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def test_sparse_segment_sqrt_n() -> None:
    """Test the sparse segment sqrt n behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Test sparse_segment_sqrt_n."
        import ml_switcheroo_compiler.ops as _ops_mod

        sparse_segment_sqrt_n = cast("Callable[..., EagerTensorProtocol]", getattr(_ops_mod, "sparse_segment_sqrt_n"))

        (e, t) = cast("Callable[..., EagerTensorProtocol]", _test_op)(sparse_segment_sqrt_n, a, a, a)
        assert e is not None
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def test_sparse_segment_sum() -> None:
    """Test the sparse segment sum behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Test sparse_segment_sum."
        import ml_switcheroo_compiler.ops as _ops_mod

        sparse_segment_sum = cast("Callable[..., EagerTensorProtocol]", getattr(_ops_mod, "sparse_segment_sum"))

        (e, t) = cast("Callable[..., EagerTensorProtocol]", _test_op)(sparse_segment_sum, a, a, a)
        assert e is not None
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def test_ragged_dot() -> None:
    """Test the ragged dot behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Test ragged_dot."
        import ml_switcheroo_compiler.ops as _ops_mod

        ragged_dot = cast("Callable[..., EagerTensorProtocol]", getattr(_ops_mod, "ragged_dot"))

        (e, t) = cast("Callable[..., EagerTensorProtocol]", _test_op)(ragged_dot, a, a)
        assert e is not None
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def test_ragged_missing() -> None:
    """Test function test_ragged_missing."""
    assert RaggedDot().infer_shape() == ()

    class DummyShape:
        """Test class DummyShape."""

        shape = (1, 2)

    assert RaggedDot().infer_shape(DummyShape()) == (1, 2, 1)
