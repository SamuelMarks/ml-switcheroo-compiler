"""Tests for test_indexing_multi_dimensional."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

# ruff: noqa: E501
from ml_switcheroo_compiler.ops.shape.indexing_advanced import (
    Argpartition,
    Argsort,
    Assign,
    AssignAdd,
    AssignSub,
    BooleanMask,
    Compress,
    Diagflat,
    DiagIndices,
    DiagIndicesFrom,
    Diagonal,
    Gather,
    GatherNd,
    InvertPermutation,
    Partition,
    Scatter,
    ScatterAdd,
    ScatterNd,
    SearchSorted,
    Select,
    Sort,
    Take,
    TakeAlongAxis,
    TensorScatterUpdate,
    TopK,
    Vdot,
    Where,
    _normalize_k,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


class MockTensor:
    """Test class MockTensor."""

    def __init__(self, shape: EagerValue) -> None:
        """Test function __init__."""
        self.shape = shape


def test_normalize_k() -> None:
    """Test function test_normalize_k."""
    assert _normalize_k(3) == 3
    assert cast("Callable[..., EagerTensorProtocol]", _normalize_k)("3") == 3

    class FakeItem:
        """Test class FakeItem."""

        def item(self) -> EagerValue:
            """Test function item."""
            return 5

    assert cast("Callable[..., EagerTensorProtocol]", _normalize_k)(cast("Callable[..., EagerTensorProtocol]", FakeItem)()) == 5

    class FakeArray:
        """Test class FakeArray."""

        def __array__(self) -> EagerValue:
            """Test function __array__."""
            return 7

    assert cast("Callable[..., EagerTensorProtocol]", _normalize_k)(cast("Callable[..., EagerTensorProtocol]", FakeArray)()) == 7
    assert cast("Callable[..., EagerTensorProtocol]", _normalize_k)("abc") == "abc"


def test_topk_infer_shape() -> None:
    """Test function test_topk_infer_shape."""
    op = TopK()
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(None) == (None,)
    t = MockTensor((2, 3))
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(t, k=2) == (2, 2)
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(t, k=None) == (2, 1)


def test_argsort_infer_shape() -> None:
    """Test function test_argsort_infer_shape."""
    op = Argsort()
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(None) == ()

    class MockTuple(tuple[EagerValue, ...]):
        """Test class MockTuple."""

        @property
        def shape(self) -> EagerValue:
            """Test function shape."""
            return (2, 3)

    t1 = MockTuple((1, 2))
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(t1) == (2, 3)
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", MockTensor)((2, 3))) == (2, 3)


def test_sort_infer_shape() -> None:
    """Test function test_sort_infer_shape."""
    op = Sort()
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", MockTensor)((2, 3))) == (2, 3)


def test_where_infer_shape() -> None:
    """Test function test_where_infer_shape."""
    op = Where()
    assert op.infer_shape() == (None,)
    assert op.infer_shape(None, (2, 3)) == (2, 3)


def test_empty_infer_shapes() -> None:
    """Test function test_empty_infer_shapes."""
    assert Gather().infer_shape() == (None,)
    assert Take().infer_shape() == (None,)
    assert TakeAlongAxis().infer_shape() == (None,)
    assert GatherNd().infer_shape() == (None,)
    assert Scatter().infer_shape() == (None,)
    assert ScatterNd().infer_shape() == (None,)
    assert ScatterAdd().infer_shape() == (None,)
    assert Vdot().infer_shape() == (None,)
    assert SearchSorted().infer_shape() == (None,)
    assert Select().infer_shape() == (None,)
    assert Assign().infer_shape() == (None,)
    assert AssignAdd().infer_shape() == (None,)
    assert AssignSub().infer_shape() == (None,)


def test_tensor_scatter_update_infer_shape() -> None:
    """Test function test_tensor_scatter_update_infer_shape."""
    op = TensorScatterUpdate()
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", MockTensor)((2, 3)), None, None) == (2, 3)


def test_argpartition_infer_shape() -> None:
    """Test function test_argpartition_infer_shape."""
    op = Argpartition()
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", MockTensor)((2, 3)), 1) == (2, 3)
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(None, 1) == ()


def test_partition_infer_shape() -> None:
    """Test function test_partition_infer_shape."""
    op = Partition()
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", MockTensor)((2, 3)), 1) == (2, 3)
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(None, 1) == ()


def test_compress_infer_shape() -> None:
    """Test function test_compress_infer_shape."""
    op = Compress()
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(None, cast("Callable[..., EagerTensorProtocol]", MockTensor)((2, 3))) == (None,)


def test_diagonal_infer_shape() -> None:
    """Test function test_diagonal_infer_shape."""
    op = Diagonal()
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", MockTensor)((2, 3))) == (None,)


def test_diagflat_infer_shape() -> None:
    """Test function test_diagflat_infer_shape."""
    op = Diagflat()
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(None) == (None, None)


def test_diag_indices_infer_shape() -> None:
    """Test function test_diag_indices_infer_shape."""
    op = DiagIndices()
    assert op.infer_shape(3) == (None,)


def test_diag_indices_from_infer_shape() -> None:
    """Test function test_diag_indices_from_infer_shape."""
    op = DiagIndicesFrom()
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", MockTensor)((2, 3))) == (None,)


def test_boolean_mask_infer_shape() -> None:
    """Test function test_boolean_mask_infer_shape."""
    op = BooleanMask()
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", MockTensor)((2, 3)), cast("Callable[..., EagerTensorProtocol]", MockTensor)((2, 3))) == (None,)


def test_invert_permutation_infer_shape() -> None:
    """Test function test_invert_permutation_infer_shape."""
    op = InvertPermutation()
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", MockTensor)((2, 3))) == (2, 3)
