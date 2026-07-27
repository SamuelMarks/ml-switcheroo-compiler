# ruff: noqa: E501
import pytest

from ml_switcheroo_compiler.ops.shape.reshape import Atleast1d, Atleast2d, Atleast3d, BroadcastInDim, BroadcastTo, Delete, ExpandDims, FillDiagonal, Flip, Fliplr, Flipud, Insert, Moveaxis, Permute, Reshape, Resize, Roll, Squeeze, Swapaxes, Transpose, _get_shape_list, _resolve_reshape_minus_one


class MockTensor:
    def __init__(self, shape=None):
        self.shape = shape
        self.dtype = "float32"
        self.device = "cpu"
        self.data = [1, 2]


class NoShape:
    pass


def test_get_shape_list():
    assert _get_shape_list(NoShape()) == []
    assert _get_shape_list(MockTensor(None)) == []
    assert _get_shape_list(MockTensor((2, 3))) == [2, 3]
    assert _get_shape_list(MockTensor(2)) == [2]
    assert _get_shape_list(MockTensor([2, 3])) == [2, 3]

    class UnrecognizedShape:
        shape = "invalid"

    assert _get_shape_list(UnrecognizedShape()) == []


def test_resolve_reshape_minus_one():
    assert _resolve_reshape_minus_one((2, 3), [1, -1, 3]) == [1, 2, 3]
    assert _resolve_reshape_minus_one((2, 3), [0, -1]) == [0, -1]


def test_transpose_infer_shape():
    op = Transpose()
    assert op.infer_shape(x=(2, 3), axes=(1, 0)) == (3, 2)
    assert op.infer_shape(x=NoShape(), axes=None) is None
    assert op._format_args("x", None) == "x"
    assert op._format_args("x", (1, 0)) == "x, (1, 0)"


def test_broadcast_to_infer_shape():
    op = BroadcastTo()
    assert op.infer_shape((2, 3), (4, 2, 3)) == (4, 2, 3)
    assert op.infer_shape("not a tuple", "not a tuple") == "not a tuple"
    with pytest.raises(ValueError, match="cannot be broadcast"):
        op.infer_shape((4, 2, 3), (2, 3))


def test_broadcast_in_dim_infer_shape():
    op = BroadcastInDim()
    assert op.infer_shape(None, (2, 3), None) == (2, 3)


def test_resize_infer_shape():
    op = Resize()
    assert op.infer_shape(NoShape(), (5, 6)) == ()
    assert op.infer_shape(MockTensor(None), (5, 6)) == ()
    assert op.infer_shape(MockTensor((2, 3, 4, 3)), (5, 6)) == (2, 5, 6, 3)


def test_atleast_dims_infer_shape():
    op1 = Atleast1d()
    op2 = Atleast2d()
    op3 = Atleast3d()
    assert op1.infer_shape(NoShape()) == (1,)
    assert op1.infer_shape(MockTensor(())) == (1,)
    assert op1.infer_shape(MockTensor((3,))) == (3,)
    assert op2.infer_shape(NoShape()) == (1, 1)
    assert op2.infer_shape(MockTensor(())) == (1, 1)
    assert op2.infer_shape(MockTensor((3,))) == (1, 3)
    assert op2.infer_shape(MockTensor((2, 3))) == (2, 3)
    assert op3.infer_shape(NoShape()) == (1, 1, 1)
    assert op3.infer_shape(MockTensor(())) == (1, 1, 1)
    assert op3.infer_shape(MockTensor((3,))) == (1, 3, 1)
    assert op3.infer_shape(MockTensor((2, 3))) == (2, 3, 1)
    assert op3.infer_shape(MockTensor((2, 3, 4))) == (2, 3, 4)


def test_delete_infer_shape():
    op = Delete()
    assert op.infer_shape(NoShape()) is None
    assert op.infer_shape(arr=MockTensor((2, 3))) == (2, 3)


def test_fill_diagonal_infer_shape():
    op = FillDiagonal()
    assert op.infer_shape(NoShape()) is None
    assert op.infer_shape(a=MockTensor((2, 3))) == (2, 3)


def test_insert_infer_shape():
    op = Insert()
    assert op.infer_shape(NoShape()) is None
    assert op.infer_shape(arr=MockTensor((2, 3))) == (2, 3)


def test_moveaxis_infer_shape():
    op = Moveaxis()
    assert op.infer_shape(NoShape()) is None
    assert op.infer_shape(MockTensor(None)) is None
    assert op.infer_shape(MockTensor((2, 3, 4)), 0, -1) == (3, 4, 2)
    assert op.infer_shape(MockTensor((2, 3, 4)), [0, 1], [-1, -2]) == (4, 3, 2)


def test_permute_infer_shape():
    op = Permute()
    assert op.infer_shape(NoShape()) is None
    assert op.infer_shape(MockTensor((2, 3, 4)), dims=None) == (4, 3, 2)
    assert op.infer_shape(MockTensor((2, 3, 4)), dims=(1, 2, 0)) == (3, 4, 2)


def test_roll_infer_shape():
    op = Roll()
    assert op.infer_shape(NoShape()) is None
    assert op.infer_shape(MockTensor((2, 3))) == (2, 3)


def test_squeeze_infer_shape():
    op = Squeeze()
    assert op.infer_shape(NoShape()) is None
    assert op.infer_shape(MockTensor((2, 1, 3))) == (2, 3)
    assert op.infer_shape(MockTensor((2, 1, 3)), axis=1) == (2, 3)
    assert op.infer_shape(MockTensor((2, 1, 3)), axis=-2) == (2, 3)


def test_swapaxes_infer_shape():
    op = Swapaxes()
    assert op.infer_shape(NoShape()) is None
    assert op.infer_shape(MockTensor((2, 3, 4)), 0, 2) == (4, 3, 2)
    assert op.infer_shape(MockTensor((2, 3, 4)), -1, -3) == (4, 3, 2)


def test_flips_infer_shape():
    assert Flip().infer_shape(NoShape()) is None
    assert Fliplr().infer_shape(NoShape()) is None
    assert Flipud().infer_shape(NoShape()) is None
    assert Flip().infer_shape(MockTensor((2, 3))) == (2, 3)
    assert Fliplr().infer_shape(MockTensor((2, 3))) == (2, 3)
    assert Flipud().infer_shape(MockTensor((2, 3))) == (2, 3)


def test_reshape_infer_shape():
    op = Reshape()
    assert op.infer_shape(MockTensor((2, 3)), (6,)) == (6,)
    assert op.infer_shape(MockTensor((2, 3)), (1, -1)) == (1, 6)
    assert op.infer_shape(NoShape(), (6,)) == (6,)
    assert op.infer_shape(MockTensor((2, 3)), "not list") == "not list"


def test_missing_reshape_classes():
    from ml_switcheroo_compiler.ops.shape.reshape import Block, C, Collapse, Diagflat, DiagIndices, DiagIndicesFrom

    op = Resize()
    assert op.infer_shape(MockTensor((2, 3)), (5, 6)) == (2, 3)
    op = ExpandDims()
    assert op.infer_shape(MockTensor(None)) == (None,)
    assert op.infer_shape(MockTensor((2, 3)), axis=None) == (2, 3)
    assert op.infer_shape(MockTensor((2, 3)), axis=0) == (1, 2, 3)
    assert op.infer_shape(MockTensor((2, 3)), axis=-1) == (2, 3, 1)
    op = Block()
    assert op.infer_shape(arrays=[1, 2]) == ()
    assert op.infer_shape(arrays=1) == ()
    assert C().infer_shape() == ()
    assert Collapse().infer_shape() == ()
    op = DiagIndices()
    assert op.infer_shape(n=3) == ((3,), (3,))
    op = DiagIndicesFrom()
    assert op.infer_shape(arr=MockTensor((2, 3))) == ((2,), (2,))
    assert op.infer_shape(NoShape()) is None
    op = Diagflat()
    assert op.infer_shape(v=MockTensor((2, 3)), k=1) == (7, 7)
    assert op.infer_shape(NoShape()) is None
