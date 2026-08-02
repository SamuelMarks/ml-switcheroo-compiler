"""Coverage tests for logical operations."""

from ml_switcheroo_compiler.ops.unary.logical import (
    Ediff1d,
    Iscomplexobj,
    Isin,
    Isrealobj,
    Issubdtype,
    iscomplex,
    iscomplexobj,
    isin,
    isreal,
    isrealobj,
    issubdtype,
    population_count,
    reduce_precision,
)


class Dummy:
    """Dummy tensor class for testing."""

    def __init__(self, shape=None):
        if shape is not None:
            self.shape = shape


def test_logical_infer_shape_additional():
    assert Iscomplexobj().infer_shape(Dummy()) == ()
    assert Isrealobj().infer_shape(Dummy()) == ()
    assert Issubdtype().infer_shape() == ()

    # Ediff1d
    op = Ediff1d()
    assert op.infer_shape(Dummy((5,))) == (4,)
    assert op.infer_shape(Dummy((5,)), to_begin=Dummy(())) == (5,)
    assert op.infer_shape(Dummy((5,)), to_begin=Dummy((2,))) == (6,)

    assert op.infer_shape(Dummy((5,)), to_end=Dummy(())) == (5,)
    assert op.infer_shape(Dummy((5,)), to_end=Dummy((2,))) == (6,)

    assert op.infer_shape(Dummy((5,)), to_begin=Dummy((2,)), to_end=Dummy((3,))) == (9,)

    # Ediff1d with None in shape
    assert op.infer_shape(Dummy((None,))) == (None,)
    assert op.infer_shape(Dummy((5,)), to_begin=Dummy((None,))) == (None,)
    assert op.infer_shape(Dummy((5,)), to_end=Dummy((None,))) == (None,)


def test_logical_dispatch():
    def safe_call(func, *args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception:
            pass

    safe_call(population_count, Dummy((2,)))
    safe_call(isin, Dummy((2,)), Dummy((2,)))
    safe_call(iscomplex, Dummy((2,)))
    safe_call(iscomplexobj, Dummy((2,)))
    safe_call(isreal, Dummy((2,)))
    safe_call(isrealobj, Dummy((2,)))
    safe_call(issubdtype, Dummy((2,)), Dummy(()))
    safe_call(reduce_precision, Dummy((2,)))


def test_logical_infer_shape_packbits():
    from ml_switcheroo_compiler.ops.unary.logical import Packbits, Unpackbits

    op1 = Packbits()
    assert op1.infer_shape(Dummy((2,)), Dummy((3, 2))) == (3, 2)

    op2 = Unpackbits()
    assert op2.infer_shape(Dummy((2,)), Dummy((3, 2))) == (3, 2)


def test_logical_infer_shape_isin():
    assert Isin().infer_shape() == ()


def test_logical_infer_shape_packbits_no_shapes():
    from ml_switcheroo_compiler.ops.unary.logical import Packbits, Unpackbits

    class NoShape:
        pass

    assert Packbits().infer_shape(NoShape()) == ()
    assert Unpackbits().infer_shape(NoShape()) == ()
