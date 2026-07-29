# ruff: noqa: E501
from ml_switcheroo_compiler.ops.unary.logical import Packbits, Unpackbits


def test_logical_infer_shape():
    assert Packbits().infer_shape() == ()
    assert Unpackbits().infer_shape() == ()

    class Dummy:
        shape = (2, 3)

    assert Packbits().infer_shape(Dummy()) == (2, 3)
    assert Unpackbits().infer_shape(Dummy()) == (2, 3)

    assert Packbits().infer_shape(Dummy(), Dummy()) == (2, 3)
    assert Unpackbits().infer_shape(Dummy(), Dummy()) == (2, 3)

    from ml_switcheroo_compiler.ops.unary.logical import Isin

    assert Isin().infer_shape() == ()
    assert Isin().infer_shape(Dummy(), Dummy()) == (2, 3)
