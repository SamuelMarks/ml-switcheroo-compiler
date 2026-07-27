# ruff: noqa: E501
from ml_switcheroo_compiler.ops.unary.special import Lbeta


class MockTensor:
    def __init__(self, shape=()):
        self.shape = shape
        self.dtype = "float32"
        self.device = "cpu"
        self.data = [1, 2]


def test_mvlgamma_infer_shape():
    op = Lbeta()
    assert op.infer_shape(MockTensor((2, 3))) == (2,)
    assert op.infer_shape(MockTensor(())) == ()


def test_missing_classes_infer_shape():
    from ml_switcheroo_compiler.ops.unary.special import CanCast, Cast, Frexp

    assert Cast().infer_shape("x") == "x"
    assert CanCast().infer_shape() == ()
    assert Frexp().infer_shape("x") == "x"
