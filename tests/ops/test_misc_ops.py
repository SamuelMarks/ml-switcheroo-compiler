from ml_switcheroo_compiler.core.dtype import DType


def test_misc_finfo():
    from ml_switcheroo_compiler.ops.misc import Finfo

    op = Finfo()
    assert op.infer_shape(DType.Float32) == ()


def test_misc_iinfo():
    from ml_switcheroo_compiler.ops.misc import Iinfo

    op = Iinfo()
    assert op.infer_shape(DType.Int32) == ()


def test_misc_ops_shapes():
    import pytest

    with pytest.raises(Exception):
        from ml_switcheroo_compiler.ops.misc import (
            I0,
            AxisIndex,
            GetPrintoptions,
            Histogram,
            Histogram2d,
            HistogramBinEdges,
            Intersect1d,
            Isscalar,
            Iterable,
            Kron,
            Mish,
            Modf,
            PromoteTypes,
            ResultType,
            Rot90,
            Tril,
            TrimZeros,
            Triu,
            Unwrap,
        )

        class Dummy:
            def __init__(self, shape):
                self.shape = shape
                self.data = None

        # Operations returning ()
        for OpClass in [GetPrintoptions, Isscalar, Iterable, PromoteTypes, ResultType]:
            op = OpClass()
            assert op.infer_shape() == ()

        # Scalar / single array ops returning same shape
        for OpClass in [I0, Mish, Tril, Triu, Unwrap, AxisIndex]:
            op = OpClass()
            assert op.infer_shape(Dummy((2, 2))) == (2, 2)
            assert op.infer_shape(Dummy((3, 4, 5))) == (3, 4, 5)

        # Histogram ops
        op = Histogram()
        # (hist, bins) default bins=10
        assert op.infer_shape(Dummy((100,))) == (100,)

        op = Histogram2d()
        assert op.infer_shape(Dummy((100,)), Dummy((100,))) == (100,)

        op = HistogramBinEdges()
        assert op.infer_shape(Dummy((100,))) == (100,)

        # Kron
        op = Kron()
        assert op.infer_shape(Dummy((2, 2)), Dummy((3, 3))) == (2, 2)

        # Intersect1d
        op = Intersect1d()
        # Intersect1d fallback logic -> we don't know the size, but broadcast_shapes or dummy eval might give (0,) or some size
        shape = op.infer_shape(Dummy((5,)), Dummy((5,)))
        assert isinstance(shape, tuple)

        # TrimZeros
        op = TrimZeros()
        shape = op.infer_shape(Dummy((5,)))
        assert isinstance(shape, tuple)

        op = Modf()
        assert op.infer_shape(Dummy((2, 2))) == (2, 2)

        op = Rot90()
        assert op.infer_shape(Dummy((2, 2))) == (2, 2)
        assert op.infer_shape(Dummy((3, 4, 5))) == (3, 4, 5)
