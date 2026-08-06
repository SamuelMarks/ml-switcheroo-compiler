def test_misc_infer_shapes_all():
    from ml_switcheroo_compiler.ops.info_and_histograms import (
        I0,
        AxisIndex,
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

    class DummyShape:
        shape = (2, 3)
        dtype = "float32"

    for cls in [
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
        AxisIndex,
    ]:
        op = cls()
        op.infer_shape()
        op.infer_shape(DummyShape())
        op.infer_shape(DummyShape(), DummyShape())


def test_misc_ops_edge_cases():
    from ml_switcheroo_compiler.ops.info_and_histograms import Histogram, Histogram2d, HistogramBinEdges, Ix, Median, Rot90, Trapezoid

    class DummyArray:
        def __init__(self, shape):
            self.shape = shape

    # Histogram edges
    h = Histogram()
    assert h.infer_shape(DummyArray((5,)), bins=DummyArray((4,))) == (3,)
    assert h.infer_shape(DummyArray((5,)), bins=5) == (5,)
    assert h.infer_shape(DummyArray((5,)), bins=None) == (10,)

    h2 = Histogram2d()
    assert h2.infer_shape(DummyArray((5,)), DummyArray((5,)), bins=[DummyArray((4,)), DummyArray((6,))]) == (3, 5)
    assert h2.infer_shape(DummyArray((5,)), DummyArray((5,)), bins=[5, 6]) == (5, 6)
    assert h2.infer_shape(DummyArray((5,)), DummyArray((5,)), bins=[None, None]) == (10, 10)
    assert h2.infer_shape(DummyArray((5,)), DummyArray((5,)), bins=None) == (10, 10)
    assert h2.infer_shape(DummyArray((5,)), DummyArray((5,)), bins=[5, 6, 7]) == (10, 10)  # branch len(bins) != 2

    hbe = HistogramBinEdges()
    assert hbe.infer_shape(DummyArray((5,)), bins=DummyArray((4,))) == (4,)
    assert hbe.infer_shape(DummyArray((5,)), bins=5) == (6,)
    assert hbe.infer_shape(DummyArray((5,)), bins=None) == (11,)

    # Ix edges
    ix = Ix()
    assert ix.infer_shape(DummyArray((5,))) == (5,)
    assert ix.infer_shape(DummyArray(())) == (1,)

    # Median edges
    med = Median()
    assert med.infer_shape(DummyArray((2, 3)), axis=0) == (3,)
    assert med.infer_shape(DummyArray((2, 3)), axis=0, keepdims=True) == (1, 3)
    assert med.infer_shape(DummyArray((2, 3)), axis=[0, 1]) == ()
    assert med.infer_shape(DummyArray((2, 3)), axis=[0, 1], keepdims=True) == (1, 1)
    assert med.infer_shape(DummyArray((2, 3)), axis=[5]) == (2, 3)  # branch ax >= len(shape)

    # Rot90 edges
    rot = Rot90()
    assert rot.infer_shape(DummyArray((2, 3)), axes=(0, 1)) == (3, 2)
    assert rot.infer_shape(DummyArray((2, 3)), axes=(1, 0)) == (3, 2)
    assert rot.infer_shape(DummyArray((2, 3)), axes=(5, 6)) == (2, 3)  # branch axes > len(shape)
    assert rot.infer_shape(DummyArray((2, 3)), axes=(0, 1, 2)) == (2, 3)  # len(axes) != 2

    # Trapezoid edges
    trap = Trapezoid()
    assert trap.infer_shape(DummyArray((2, 3)), axis=0) == (3,)
    assert trap.infer_shape(DummyArray((2, 3)), axis=1) == (2,)
    assert trap.infer_shape(DummyArray((2, 3)), axis=5) == (2, 3)  # branch axis > len(shape)
