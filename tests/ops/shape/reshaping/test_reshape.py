"""Tests for test_reshape."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

# ruff: noqa: E501
import pytest

from ml_switcheroo_compiler.ops.shape.reshape import Atleast1d, Atleast2d, Atleast3d, BroadcastInDim, BroadcastTo, Delete, ExpandDims, FillDiagonal, Flip, Fliplr, Flipud, Insert, Moveaxis, Permute, Reshape, Resize, Roll, Squeeze, Swapaxes, Transpose, _get_shape_list, _resolve_reshape_minus_one

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerValue


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
    assert cast("EagerCallableProtocol", _get_shape_list)(cast("EagerCallableProtocol", NoShape)()) == []
    assert cast("EagerCallableProtocol", _get_shape_list)(cast("EagerCallableProtocol", MockTensor)(None)) == []
    assert cast("EagerCallableProtocol", _get_shape_list)(cast("EagerCallableProtocol", MockTensor)((2, 3))) == [2, 3]
    assert cast("EagerCallableProtocol", _get_shape_list)(cast("EagerCallableProtocol", MockTensor)(2)) == [2]
    assert cast("EagerCallableProtocol", _get_shape_list)(cast("EagerCallableProtocol", MockTensor)([2, 3])) == [2, 3]

    class UnrecognizedShape:
        """Test class UnrecognizedShape."""

        shape = "invalid"

    assert cast("EagerCallableProtocol", _get_shape_list)(cast("EagerCallableProtocol", UnrecognizedShape)()) == []


def test_resolve_reshape_minus_one() -> None:
    """Test function test_resolve_reshape_minus_one."""
    assert _resolve_reshape_minus_one((2, 3), [1, -1, 3]) == [1, 2, 3]
    assert _resolve_reshape_minus_one((2, 3), [0, -1]) == [0, -1]


def test_transpose_infer_shape() -> None:
    """Test function test_transpose_infer_shape."""
    op = Transpose()
    assert op.infer_shape(x=(2, 3), axes=(1, 0)) == (3, 2)
    assert cast("EagerCallableProtocol", op.infer_shape)(x=cast("EagerCallableProtocol", NoShape)(), axes=None) is None
    assert op._format_args("x", None) == "x"
    assert op._format_args("x", (1, 0)) == "x, (1, 0)"


def test_broadcast_to_infer_shape() -> None:
    """Test function test_broadcast_to_infer_shape."""
    op = BroadcastTo()
    assert cast("EagerCallableProtocol", op.infer_shape)((2, 3), (4, 2, 3)) == (4, 2, 3)
    assert cast("EagerCallableProtocol", op.infer_shape)("not a tuple", "not a tuple") == "not a tuple"
    with pytest.raises(ValueError, match="cannot be broadcast"):
        cast("EagerCallableProtocol", op.infer_shape)((4, 2, 3), (2, 3))


def test_broadcast_in_dim_infer_shape() -> None:
    """Test function test_broadcast_in_dim_infer_shape."""
    op = BroadcastInDim()
    assert cast("EagerCallableProtocol", op.infer_shape)(None, (2, 3), None) == (2, 3)


def test_resize_infer_shape() -> None:
    """Test function test_resize_infer_shape."""
    op = Resize()
    assert cast("EagerCallableProtocol", op.infer_shape)(cast("EagerCallableProtocol", NoShape)(), (5, 6)) == ()
    assert cast("EagerCallableProtocol", op.infer_shape)(cast("EagerCallableProtocol", MockTensor)(None), (5, 6)) == ()
    assert cast("EagerCallableProtocol", op.infer_shape)(cast("EagerCallableProtocol", MockTensor)((2, 3, 4, 3)), (5, 6)) == (2, 5, 6, 3)


def test_atleast_dims_infer_shape() -> None:
    """Test function test_atleast_dims_infer_shape."""
    op1 = Atleast1d()
    op2 = Atleast2d()
    op3 = Atleast3d()
    assert cast("EagerCallableProtocol", op1.infer_shape)(cast("EagerCallableProtocol", NoShape)()) == (1,)
    assert cast("EagerCallableProtocol", op1.infer_shape)(cast("EagerCallableProtocol", MockTensor)(())) == (1,)
    assert cast("EagerCallableProtocol", op1.infer_shape)(cast("EagerCallableProtocol", MockTensor)((3,))) == (3,)
    assert cast("EagerCallableProtocol", op2.infer_shape)(cast("EagerCallableProtocol", NoShape)()) == (1, 1)
    assert cast("EagerCallableProtocol", op2.infer_shape)(cast("EagerCallableProtocol", MockTensor)(())) == (1, 1)
    assert cast("EagerCallableProtocol", op2.infer_shape)(cast("EagerCallableProtocol", MockTensor)((3,))) == (1, 3)
    assert cast("EagerCallableProtocol", op2.infer_shape)(cast("EagerCallableProtocol", MockTensor)((2, 3))) == (2, 3)
    assert cast("EagerCallableProtocol", op3.infer_shape)(cast("EagerCallableProtocol", NoShape)()) == (1, 1, 1)
    assert cast("EagerCallableProtocol", op3.infer_shape)(cast("EagerCallableProtocol", MockTensor)(())) == (1, 1, 1)
    assert cast("EagerCallableProtocol", op3.infer_shape)(cast("EagerCallableProtocol", MockTensor)((3,))) == (1, 3, 1)
    assert cast("EagerCallableProtocol", op3.infer_shape)(cast("EagerCallableProtocol", MockTensor)((2, 3))) == (2, 3, 1)
    assert cast("EagerCallableProtocol", op3.infer_shape)(cast("EagerCallableProtocol", MockTensor)((2, 3, 4))) == (2, 3, 4)


def test_delete_infer_shape() -> None:
    """Test function test_delete_infer_shape."""
    op = Delete()
    assert cast("EagerCallableProtocol", op.infer_shape)(cast("EagerCallableProtocol", NoShape)()) is None
    assert cast("EagerCallableProtocol", op.infer_shape)(arr=cast("EagerCallableProtocol", MockTensor)((2, 3))) == (2, 3)


def test_fill_diagonal_infer_shape() -> None:
    """Test function test_fill_diagonal_infer_shape."""
    op = FillDiagonal()
    assert cast("EagerCallableProtocol", op.infer_shape)(cast("EagerCallableProtocol", NoShape)()) is None
    assert cast("EagerCallableProtocol", op.infer_shape)(a=cast("EagerCallableProtocol", MockTensor)((2, 3))) == (2, 3)


def test_insert_infer_shape() -> None:
    """Test function test_insert_infer_shape."""
    op = Insert()
    assert cast("EagerCallableProtocol", op.infer_shape)(cast("EagerCallableProtocol", NoShape)()) is None
    assert cast("EagerCallableProtocol", op.infer_shape)(arr=cast("EagerCallableProtocol", MockTensor)((2, 3))) == (2, 3)


def test_moveaxis_infer_shape() -> None:
    """Test function test_moveaxis_infer_shape."""
    op = Moveaxis()
    assert cast("EagerCallableProtocol", op.infer_shape)(cast("EagerCallableProtocol", NoShape)()) is None
    assert cast("EagerCallableProtocol", op.infer_shape)(cast("EagerCallableProtocol", MockTensor)(None)) is None
    assert cast("EagerCallableProtocol", op.infer_shape)(cast("EagerCallableProtocol", MockTensor)((2, 3, 4)), 0, -1) == (3, 4, 2)
    assert cast("EagerCallableProtocol", op.infer_shape)(cast("EagerCallableProtocol", MockTensor)((2, 3, 4)), [0, 1], [-1, -2]) == (4, 3, 2)


def test_permute_infer_shape() -> None:
    """Test function test_permute_infer_shape."""
    op = Permute()
    assert cast("EagerCallableProtocol", op.infer_shape)(cast("EagerCallableProtocol", NoShape)()) is None
    assert cast("EagerCallableProtocol", op.infer_shape)(cast("EagerCallableProtocol", MockTensor)((2, 3, 4)), dims=None) == (4, 3, 2)
    assert cast("EagerCallableProtocol", op.infer_shape)(cast("EagerCallableProtocol", MockTensor)((2, 3, 4)), dims=(1, 2, 0)) == (3, 4, 2)


def test_roll_infer_shape() -> None:
    """Test function test_roll_infer_shape."""
    op = Roll()
    assert cast("EagerCallableProtocol", op.infer_shape)(cast("EagerCallableProtocol", NoShape)()) is None
    assert cast("EagerCallableProtocol", op.infer_shape)(cast("EagerCallableProtocol", MockTensor)((2, 3))) == (2, 3)


def test_squeeze_infer_shape() -> None:
    """Test function test_squeeze_infer_shape."""
    op = Squeeze()
    assert cast("EagerCallableProtocol", op.infer_shape)(cast("EagerCallableProtocol", NoShape)()) is None
    assert cast("EagerCallableProtocol", op.infer_shape)(cast("EagerCallableProtocol", MockTensor)((2, 1, 3))) == (2, 3)
    assert cast("EagerCallableProtocol", op.infer_shape)(cast("EagerCallableProtocol", MockTensor)((2, 1, 3)), axis=1) == (2, 3)
    assert cast("EagerCallableProtocol", op.infer_shape)(cast("EagerCallableProtocol", MockTensor)((2, 1, 3)), axis=-2) == (2, 3)


def test_swapaxes_infer_shape() -> None:
    """Test function test_swapaxes_infer_shape."""
    op = Swapaxes()
    assert cast("EagerCallableProtocol", op.infer_shape)(cast("EagerCallableProtocol", NoShape)()) is None
    assert cast("EagerCallableProtocol", op.infer_shape)(cast("EagerCallableProtocol", MockTensor)((2, 3, 4)), 0, 2) == (4, 3, 2)
    assert cast("EagerCallableProtocol", op.infer_shape)(cast("EagerCallableProtocol", MockTensor)((2, 3, 4)), -1, -3) == (4, 3, 2)


def test_flips_infer_shape() -> None:
    """Test function test_flips_infer_shape."""
    assert cast("EagerCallableProtocol", Flip)().infer_shape(cast("EagerCallableProtocol", NoShape)()) is None
    assert cast("EagerCallableProtocol", Fliplr)().infer_shape(cast("EagerCallableProtocol", NoShape)()) is None
    assert cast("EagerCallableProtocol", Flipud)().infer_shape(cast("EagerCallableProtocol", NoShape)()) is None
    assert cast("EagerCallableProtocol", Flip)().infer_shape(cast("EagerCallableProtocol", MockTensor)((2, 3))) == (2, 3)
    assert cast("EagerCallableProtocol", Fliplr)().infer_shape(cast("EagerCallableProtocol", MockTensor)((2, 3))) == (2, 3)
    assert cast("EagerCallableProtocol", Flipud)().infer_shape(cast("EagerCallableProtocol", MockTensor)((2, 3))) == (2, 3)


def test_reshape_infer_shape() -> None:
    """Test function test_reshape_infer_shape."""
    op = Reshape()
    assert cast("EagerCallableProtocol", op.infer_shape)(cast("EagerCallableProtocol", MockTensor)((2, 3)), (6,)) == (6,)
    assert cast("EagerCallableProtocol", op.infer_shape)(cast("EagerCallableProtocol", MockTensor)((2, 3)), (1, -1)) == (1, 6)
    assert cast("EagerCallableProtocol", op.infer_shape)(cast("EagerCallableProtocol", NoShape)(), (6,)) == (6,)
    assert cast("EagerCallableProtocol", op.infer_shape)(cast("EagerCallableProtocol", MockTensor)((2, 3)), "not list") == "not list"


def test_missing_reshape_classes() -> None:
    """Test function test_missing_reshape_classes."""
    from ml_switcheroo_compiler.ops.shape.reshape import Block, C, Collapse, Diagflat, DiagIndices, DiagIndicesFrom

    op = Resize()
    assert cast("EagerCallableProtocol", op.infer_shape)(cast("EagerCallableProtocol", MockTensor)((2, 3)), (5, 6)) == (2, 3)
    op = cast("Resize", ExpandDims())
    assert cast("EagerCallableProtocol", op.infer_shape)(cast("EagerCallableProtocol", MockTensor)(None)) == (None,)
    assert cast("EagerCallableProtocol", op.infer_shape)(cast("EagerCallableProtocol", MockTensor)((2, 3)), axis=None) == (2, 3)
    assert cast("EagerCallableProtocol", op.infer_shape)(cast("EagerCallableProtocol", MockTensor)((2, 3)), axis=0) == (1, 2, 3)
    assert cast("EagerCallableProtocol", op.infer_shape)(cast("EagerCallableProtocol", MockTensor)((2, 3)), axis=-1) == (2, 3, 1)
    op = cast("Resize", Block())
    assert cast("EagerCallableProtocol", op.infer_shape)(arrays=[1, 2]) == ()
    assert cast("EagerCallableProtocol", op.infer_shape)(arrays=1) == ()
    assert C().infer_shape() == ()
    assert Collapse().infer_shape() == ()
    op = cast("Resize", DiagIndices())
    assert cast("EagerCallableProtocol", op.infer_shape)(n=3) == ((3,), (3,))
    op = cast("Resize", DiagIndicesFrom())
    assert cast("EagerCallableProtocol", op.infer_shape)(arr=cast("EagerCallableProtocol", MockTensor)((2, 3))) == ((2,), (2,))
    assert cast("EagerCallableProtocol", op.infer_shape)(cast("EagerCallableProtocol", NoShape)()) is None
    op = cast("Resize", Diagflat())
    assert cast("EagerCallableProtocol", op.infer_shape)(v=cast("EagerCallableProtocol", MockTensor)((2, 3)), k=1) == (7, 7)
    assert cast("EagerCallableProtocol", op.infer_shape)(cast("EagerCallableProtocol", NoShape)()) is None
