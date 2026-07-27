from ml_switcheroo_compiler.ops.misc import (
    I0,
    Finfo,
    GetPrintoptions,
    Gradient,
    Histogram,
    Histogram2d,
    HistogramBinEdges,
    Histogramdd,
    Iinfo,
    Indices,
    Infeed,
    Interp,
    Intersect1d,
    Isscalar,
    Iterable,
    Ix,
    Kron,
    MaskIndices,
    Median,
    Mgrid,
    Mish,
    Modf,
    Ogrid,
    Piecewise,
    PromoteTypes,
    R,
    ResultType,
    Rot90,
    Trapezoid,
    Tri,
    Tril,
    TrimZeros,
    Triu,
    Unwrap,
    Vander,
    Vectorize,
)


def test_misc_stubs():
    for OpClass in [
        Finfo,
        Iinfo,
        GetPrintoptions,
        Gradient,
        Histogram,
        Histogram2d,
        HistogramBinEdges,
        Histogramdd,
        I0,
        Indices,
        Infeed,
        Interp,
        Intersect1d,
        Isscalar,
        Iterable,
        Ix,
        Kron,
        MaskIndices,
        Median,
        Mgrid,
        Mish,
        Modf,
        Ogrid,
        Piecewise,
        PromoteTypes,
        R,
        ResultType,
        Rot90,
        Trapezoid,
        Tri,
        Tril,
        TrimZeros,
        Triu,
        Unwrap,
        Vander,
        Vectorize,
    ]:
        op = OpClass()

        class Dummy:
            shape = (2, 2)

        try:
            op.infer_shape(Dummy())
        except Exception:
            pass
        try:
            op.infer_shape()
        except Exception:
            pass
        try:
            op.infer_shape(Dummy(), Dummy())
        except Exception:
            pass
