"""Tests for test_indexing_multi_dimensional."""

from __future__ import annotations

import typing

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
from tests.testing_utils import MockTestStub

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


class MockTensor(MockTestStub):
    """Test class MockTensor."""

    def __init__(self, shape: BackendArg) -> None:
        """Test function __init__.

        Args:
            shape: Test parameter `shape`.
        """
        self.shape = shape


def test_normalize_k() -> None:
    """Test function test_normalize_k."""
    assert _normalize_k(3) == 3
    assert _normalize_k("3") == 3

    class FakeItem(MockTestStub):
        """Test class FakeItem."""

        def item(self) -> BackendArg:
            """Test function item.

            Returns:
                BackendArg: Computed output value from `item`.
            """
            return 5

    assert _normalize_k(FakeItem()) == 5

    class FakeArray(MockTestStub):
        """Test class FakeArray."""

        def __array__(self) -> BackendArg:
            """Test function __array__.

            Returns:
                BackendArg: Computed output value from `__array__`.
            """
            return 7

    assert _normalize_k(FakeArray()) == 7
    assert _normalize_k("abc") == "abc"


def test_topk_infer_shape() -> None:
    """Test function test_topk_infer_shape."""
    op = TopK()
    assert op.infer_shape(None) == (None,)
    t = MockTensor((2, 3))
    assert op.infer_shape(t, k=2) == (2, 2)
    assert op.infer_shape(t, k=None) == (2, 1)


def test_argsort_infer_shape() -> None:
    """Test function test_argsort_infer_shape."""
    op = Argsort()
    assert op.infer_shape(None) == ()

    class MockTuple(tuple[int, ...]):
        """Test class MockTuple."""

        @property
        def shape(self) -> BackendArg:
            """Test function shape.

            Returns:
                BackendArg: Computed output value from `shape`.
            """
            return (2, 3)

    t1 = MockTuple((1, 2))
    assert op.infer_shape(t1) == (2, 3)
    assert op.infer_shape(MockTensor((2, 3))) == (2, 3)


def test_sort_infer_shape() -> None:
    """Test function test_sort_infer_shape."""
    op = Sort()
    assert op.infer_shape(MockTensor((2, 3))) == (2, 3)


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
    assert op.infer_shape(MockTensor((2, 3)), None, None) == (2, 3)


def test_argpartition_infer_shape() -> None:
    """Test function test_argpartition_infer_shape."""
    op = Argpartition()
    assert op.infer_shape(MockTensor((2, 3)), 1) == (2, 3)
    assert op.infer_shape(None, 1) == ()


def test_partition_infer_shape() -> None:
    """Test function test_partition_infer_shape."""
    op = Partition()
    assert op.infer_shape(MockTensor((2, 3)), 1) == (2, 3)
    assert op.infer_shape(None, 1) == ()


def test_compress_infer_shape() -> None:
    """Test function test_compress_infer_shape."""
    op = Compress()
    assert op.infer_shape(None, MockTensor((2, 3))) == (None,)


def test_diagonal_infer_shape() -> None:
    """Test function test_diagonal_infer_shape."""
    op = Diagonal()
    assert op.infer_shape(MockTensor((2, 3))) == (None,)


def test_diagflat_infer_shape() -> None:
    """Test function test_diagflat_infer_shape."""
    op = Diagflat()
    assert op.infer_shape(None) == (None, None)


def test_diag_indices_infer_shape() -> None:
    """Test function test_diag_indices_infer_shape."""
    op = DiagIndices()
    assert op.infer_shape(3) == (None,)


def test_diag_indices_from_infer_shape() -> None:
    """Test function test_diag_indices_from_infer_shape."""
    op = DiagIndicesFrom()
    assert op.infer_shape(MockTensor((2, 3))) == (None,)


def test_boolean_mask_infer_shape() -> None:
    """Test function test_boolean_mask_infer_shape."""
    op = BooleanMask()
    assert op.infer_shape(MockTensor((2, 3)), MockTensor((2, 3))) == (None,)


def test_invert_permutation_infer_shape() -> None:
    """Test function test_invert_permutation_infer_shape."""
    op = InvertPermutation()
    assert op.infer_shape(MockTensor((2, 3))) == (2, 3)
