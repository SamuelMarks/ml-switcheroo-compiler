# ruff: noqa: E501
from ml_switcheroo_compiler.ops.unary.logical import Packbits, Unpackbits


def test_logical_infer_shape():
    assert Packbits().infer_shape() == ()
    assert Unpackbits().infer_shape() == ()
    assert Packbits().infer_shape((2, 3)) == (2, 3)
    assert Unpackbits().infer_shape((2, 3)) == (2, 3)
