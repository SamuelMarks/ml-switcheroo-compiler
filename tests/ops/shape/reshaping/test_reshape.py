"""Tests for test_reshape."""

from __future__ import annotations

import typing

# ruff: noqa: E501
import pytest

from ml_switcheroo_compiler.ops.shape.reshape import Atleast1d, Atleast2d, Atleast3d, BroadcastInDim, BroadcastTo, Delete, ExpandDims, FillDiagonal, Flip, Fliplr, Flipud, Insert, Moveaxis, Permute, Reshape, Resize, Roll, Squeeze, Swapaxes, Transpose, _get_shape_list, _resolve_reshape_minus_one
from tests.testing_utils import as_callable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


class MockTensor:
    """Test class MockTensor."""

    def __init__(self, shape: EagerValue = None) -> None:
        """Test function __init__.

        Args:
            shape: Test parameter `shape`.
        """
        self.shape = shape
        self.dtype = "float32"
        self.device = "cpu"
        self.data = [1, 2]


class NoShape:
    """Test class NoShape."""

    pass


def test_get_shape_list() -> None:
    """Test function test_get_shape_list."""
    assert as_callable(_get_shape_list)(as_callable(NoShape)()) == []
    assert as_callable(_get_shape_list)(as_callable(MockTensor)(None)) == []
    assert as_callable(_get_shape_list)(as_callable(MockTensor)((2, 3))) == [2, 3]
    assert as_callable(_get_shape_list)(as_callable(MockTensor)(2)) == [2]
    assert as_callable(_get_shape_list)(as_callable(MockTensor)([2, 3])) == [2, 3]

    class UnrecognizedShape:
        """Test class UnrecognizedShape."""

        shape = "invalid"

    assert as_callable(_get_shape_list)(as_callable(UnrecognizedShape)()) == []


def test_resolve_reshape_minus_one() -> None:
    """Test function test_resolve_reshape_minus_one."""
    assert _resolve_reshape_minus_one((2, 3), [1, -1, 3]) == [1, 2, 3]
    assert _resolve_reshape_minus_one((2, 3), [0, -1]) == [0, -1]


def test_transpose_infer_shape() -> None:
    """Test function test_transpose_infer_shape."""
    op = Transpose()
    assert op.infer_shape(x=(2, 3), axes=(1, 0)) == (3, 2)
    assert as_callable(op.infer_shape)(x=as_callable(NoShape)(), axes=None) is None
    assert op._format_args("x", None) == "x"
    assert op._format_args("x", (1, 0)) == "x, (1, 0)"


def test_broadcast_to_infer_shape() -> None:
    """Test function test_broadcast_to_infer_shape."""
    op = BroadcastTo()
    assert as_callable(op.infer_shape)((2, 3), (4, 2, 3)) == (4, 2, 3)
    assert as_callable(op.infer_shape)("not a tuple", "not a tuple") == "not a tuple"
    with pytest.raises(ValueError, match="cannot be broadcast"):
        as_callable(op.infer_shape)((4, 2, 3), (2, 3))


def test_broadcast_in_dim_infer_shape() -> None:
    """Test function test_broadcast_in_dim_infer_shape."""
    op = BroadcastInDim()
    assert as_callable(op.infer_shape)(None, (2, 3), None) == (2, 3)


def test_resize_infer_shape() -> None:
    """Test function test_resize_infer_shape."""
    op = Resize()
    assert as_callable(op.infer_shape)(as_callable(NoShape)(), (5, 6)) == ()
    assert as_callable(op.infer_shape)(as_callable(MockTensor)(None), (5, 6)) == ()
    assert as_callable(op.infer_shape)(as_callable(MockTensor)((2, 3, 4, 3)), (5, 6)) == (2, 5, 6, 3)


def test_atleast_dims_infer_shape() -> None:
    """Test function test_atleast_dims_infer_shape."""
    op1 = Atleast1d()
    op2 = Atleast2d()
    op3 = Atleast3d()
    assert as_callable(op1.infer_shape)(as_callable(NoShape)()) == (1,)
    assert as_callable(op1.infer_shape)(as_callable(MockTensor)(())) == (1,)
    assert as_callable(op1.infer_shape)(as_callable(MockTensor)((3,))) == (3,)
    assert as_callable(op2.infer_shape)(as_callable(NoShape)()) == (1, 1)
    assert as_callable(op2.infer_shape)(as_callable(MockTensor)(())) == (1, 1)
    assert as_callable(op2.infer_shape)(as_callable(MockTensor)((3,))) == (1, 3)
    assert as_callable(op2.infer_shape)(as_callable(MockTensor)((2, 3))) == (2, 3)
    assert as_callable(op3.infer_shape)(as_callable(NoShape)()) == (1, 1, 1)
    assert as_callable(op3.infer_shape)(as_callable(MockTensor)(())) == (1, 1, 1)
    assert as_callable(op3.infer_shape)(as_callable(MockTensor)((3,))) == (1, 3, 1)
    assert as_callable(op3.infer_shape)(as_callable(MockTensor)((2, 3))) == (2, 3, 1)
    assert as_callable(op3.infer_shape)(as_callable(MockTensor)((2, 3, 4))) == (2, 3, 4)


def test_delete_infer_shape() -> None:
    """Test function test_delete_infer_shape."""
    op = Delete()
    assert as_callable(op.infer_shape)(as_callable(NoShape)()) is None
    assert as_callable(op.infer_shape)(arr=as_callable(MockTensor)((2, 3))) == (2, 3)


def test_fill_diagonal_infer_shape() -> None:
    """Test function test_fill_diagonal_infer_shape."""
    op = FillDiagonal()
    assert as_callable(op.infer_shape)(as_callable(NoShape)()) is None
    assert as_callable(op.infer_shape)(a=as_callable(MockTensor)((2, 3))) == (2, 3)


def test_insert_infer_shape() -> None:
    """Test function test_insert_infer_shape."""
    op = Insert()
    assert as_callable(op.infer_shape)(as_callable(NoShape)()) is None
    assert as_callable(op.infer_shape)(arr=as_callable(MockTensor)((2, 3))) == (2, 3)


def test_moveaxis_infer_shape() -> None:
    """Test function test_moveaxis_infer_shape."""
    op = Moveaxis()
    assert as_callable(op.infer_shape)(as_callable(NoShape)()) is None
    assert as_callable(op.infer_shape)(as_callable(MockTensor)(None)) is None
    assert as_callable(op.infer_shape)(as_callable(MockTensor)((2, 3, 4)), 0, -1) == (3, 4, 2)
    assert as_callable(op.infer_shape)(as_callable(MockTensor)((2, 3, 4)), [0, 1], [-1, -2]) == (4, 3, 2)


def test_permute_infer_shape() -> None:
    """Test function test_permute_infer_shape."""
    op = Permute()
    assert as_callable(op.infer_shape)(as_callable(NoShape)()) is None
    assert as_callable(op.infer_shape)(as_callable(MockTensor)((2, 3, 4)), dims=None) == (4, 3, 2)
    assert as_callable(op.infer_shape)(as_callable(MockTensor)((2, 3, 4)), dims=(1, 2, 0)) == (3, 4, 2)


def test_roll_infer_shape() -> None:
    """Test function test_roll_infer_shape."""
    op = Roll()
    assert as_callable(op.infer_shape)(as_callable(NoShape)()) is None
    assert as_callable(op.infer_shape)(as_callable(MockTensor)((2, 3))) == (2, 3)


def test_squeeze_infer_shape() -> None:
    """Test function test_squeeze_infer_shape."""
    op = Squeeze()
    assert as_callable(op.infer_shape)(as_callable(NoShape)()) is None
    assert as_callable(op.infer_shape)(as_callable(MockTensor)((2, 1, 3))) == (2, 3)
    assert as_callable(op.infer_shape)(as_callable(MockTensor)((2, 1, 3)), axis=1) == (2, 3)
    assert as_callable(op.infer_shape)(as_callable(MockTensor)((2, 1, 3)), axis=-2) == (2, 3)


def test_swapaxes_infer_shape() -> None:
    """Test function test_swapaxes_infer_shape."""
    op = Swapaxes()
    assert as_callable(op.infer_shape)(as_callable(NoShape)()) is None
    assert as_callable(op.infer_shape)(as_callable(MockTensor)((2, 3, 4)), 0, 2) == (4, 3, 2)
    assert as_callable(op.infer_shape)(as_callable(MockTensor)((2, 3, 4)), -1, -3) == (4, 3, 2)


def test_flips_infer_shape() -> None:
    """Test function test_flips_infer_shape."""
    assert as_callable(Flip)().infer_shape(as_callable(NoShape)()) is None
    assert as_callable(Fliplr)().infer_shape(as_callable(NoShape)()) is None
    assert as_callable(Flipud)().infer_shape(as_callable(NoShape)()) is None
    assert as_callable(Flip)().infer_shape(as_callable(MockTensor)((2, 3))) == (2, 3)
    assert as_callable(Fliplr)().infer_shape(as_callable(MockTensor)((2, 3))) == (2, 3)
    assert as_callable(Flipud)().infer_shape(as_callable(MockTensor)((2, 3))) == (2, 3)


def test_reshape_infer_shape() -> None:
    """Test function test_reshape_infer_shape."""
    op = Reshape()
    assert as_callable(op.infer_shape)(as_callable(MockTensor)((2, 3)), (6,)) == (6,)
    assert as_callable(op.infer_shape)(as_callable(MockTensor)((2, 3)), (1, -1)) == (1, 6)
    assert as_callable(op.infer_shape)(as_callable(NoShape)(), (6,)) == (6,)
    assert as_callable(op.infer_shape)(as_callable(MockTensor)((2, 3)), "not list") == "not list"


def test_missing_reshape_classes() -> None:
    """Test function test_missing_reshape_classes."""
    from ml_switcheroo_compiler.ops.shape.reshape import Block, C, Collapse, Diagflat, DiagIndices, DiagIndicesFrom

    op = Resize()
    assert as_callable(op.infer_shape)(as_callable(MockTensor)((2, 3)), (5, 6)) == (2, 3)
    op = typing.cast("Resize", ExpandDims())
    assert as_callable(op.infer_shape)(as_callable(MockTensor)(None)) == (None,)
    assert as_callable(op.infer_shape)(as_callable(MockTensor)((2, 3)), axis=None) == (2, 3)
    assert as_callable(op.infer_shape)(as_callable(MockTensor)((2, 3)), axis=0) == (1, 2, 3)
    assert as_callable(op.infer_shape)(as_callable(MockTensor)((2, 3)), axis=-1) == (2, 3, 1)
    op = typing.cast("Resize", Block())
    assert as_callable(op.infer_shape)(arrays=[1, 2]) == ()
    assert as_callable(op.infer_shape)(arrays=1) == ()
    assert C().infer_shape() == ()
    assert Collapse().infer_shape() == ()
    op = typing.cast("Resize", DiagIndices())
    assert as_callable(op.infer_shape)(n=3) == ((3,), (3,))
    op = typing.cast("Resize", DiagIndicesFrom())
    assert as_callable(op.infer_shape)(arr=as_callable(MockTensor)((2, 3))) == ((2,), (2,))
    assert as_callable(op.infer_shape)(as_callable(NoShape)()) is None
    op = typing.cast("Resize", Diagflat())
    assert as_callable(op.infer_shape)(v=as_callable(MockTensor)((2, 3)), k=1) == (7, 7)
    assert as_callable(op.infer_shape)(as_callable(NoShape)()) is None
