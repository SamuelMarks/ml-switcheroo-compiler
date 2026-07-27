# ruff: noqa: E501
from ml_switcheroo_compiler.ops.shape.concat import Append, Argwhere, ColumnStack, Concatenate, Dsplit, Dstack, Hsplit, Hstack, RowStack, Split, Stack, Vsplit, Vstack


class MockTensor:
    def __init__(self, shape):
        self.shape = shape


def test_concatenate_infer_shape():
    op = Concatenate()
    assert op.infer_shape(MockTensor((2, 3))) == ()
    assert op.infer_shape([]) == ()
    shapes = [(2, 3), (4, 3)]
    assert op.infer_shape(shapes, axis=0) == (6, 3)
    shapes2 = [(2, 3), (4,)]
    assert op.infer_shape(shapes2, axis=0) == (6, 3)


def test_stack_infer_shape():
    assert Stack().infer_shape() == ()


def test_split_ops_infer_shape():
    t = MockTensor((2, 3))
    for OpClass in [Split, Hsplit, Vsplit, Dsplit]:
        op = OpClass()
        assert op.infer_shape(t) == (2, 3)
        assert op.infer_shape() == ()


def test_hstack_infer_shape():
    op = Hstack()
    assert op.infer_shape(MockTensor((2, 3))) == ()
    assert op.infer_shape([]) == ()
    assert op.infer_shape([MockTensor((2,)), MockTensor((3,))]) == (5,)
    assert op.infer_shape([MockTensor((2, 3)), MockTensor((2, 4))]) == (2, 7)


def test_vstack_infer_shape():
    op = Vstack()
    assert op.infer_shape(MockTensor((2, 3))) == ()
    assert op.infer_shape([]) == ()
    assert op.infer_shape([MockTensor((3,)), MockTensor((3,))]) == (2, 3)
    assert op.infer_shape([MockTensor((2, 3)), MockTensor((4, 3))]) == (6, 3)


def test_dstack_infer_shape():
    op = Dstack()
    assert op.infer_shape(MockTensor((2, 3))) == ()
    assert op.infer_shape([]) == ()
    assert op.infer_shape([MockTensor((3,)), MockTensor((3,))]) == (1, 3, 2)
    assert op.infer_shape([MockTensor((2, 3)), MockTensor((2, 3))]) == (2, 3, 2)
    assert op.infer_shape([MockTensor((2, 3, 4)), MockTensor((2, 3, 5))]) == (2, 3, 9)


def test_columnstack_infer_shape():
    op = ColumnStack()
    assert op.infer_shape(MockTensor((2, 3))) == ()
    assert op.infer_shape([]) == ()
    assert op.infer_shape([MockTensor((3,)), MockTensor((3,))]) == (3, 2)
    assert op.infer_shape([MockTensor((2, 3)), MockTensor((2, 4))]) == (2, 7)


def test_rowstack_infer_shape():
    assert RowStack().infer_shape() == ()


def test_argwhere_infer_shape():
    op = Argwhere()
    assert op.infer_shape((2, 3)) == (None, 2)
    assert op.infer_shape(MockTensor((2, 3))) == (None, None)


def test_append_infer_shape():
    op = Append()
    assert op.infer_shape() == ()
    assert op.infer_shape([1, 2]) == ()
    t1 = MockTensor((2, 3))
    t2 = MockTensor((4, 3))
    assert op.infer_shape(t1, t2, axis=0) == (6, 3)
    t3 = MockTensor((2, 3))
    assert op.infer_shape(t1, t3, axis=1) == (2, 6)
    assert op.infer_shape(t1, t2) == (18,)
    assert op.infer_shape(t1) == (7,)
    assert op.infer_shape(t1, axis=0) == (3, 3)
