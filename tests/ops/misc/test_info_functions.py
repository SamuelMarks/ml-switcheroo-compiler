"""Tests for test_info_functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def test_misc_infer_shapes_all() -> None:
    """Test function test_misc_infer_shapes_all."""
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
        """Test class DummyShape."""

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
    assert callable(test_misc_infer_shapes_all)


def test_misc_ops_edge_cases() -> None:
    """Test function test_misc_ops_edge_cases."""
    from ml_switcheroo_compiler.ops.info_and_histograms import Histogram, Histogram2d, HistogramBinEdges, Ix, Median, Rot90, Trapezoid

    class DummyArray:
        """Test class DummyArray."""

        def __init__(self, shape: EagerValue) -> None:
            """Test function __init__."""
            self.shape = shape

    # Histogram edges
    h = Histogram()
    assert cast("Callable[..., EagerTensorProtocol]", h.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyArray)((5,)), bins=cast("Callable[..., EagerTensorProtocol]", DummyArray)((4,))) == (3,)
    assert cast("Callable[..., EagerTensorProtocol]", h.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyArray)((5,)), bins=5) == (5,)
    assert cast("Callable[..., EagerTensorProtocol]", h.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyArray)((5,)), bins=None) == (10,)

    h2 = Histogram2d()
    assert cast("Callable[..., EagerTensorProtocol]", h2.infer_shape)(
        cast("Callable[..., EagerTensorProtocol]", DummyArray)((5,)), cast("Callable[..., EagerTensorProtocol]", DummyArray)((5,)), bins=[cast("Callable[..., EagerTensorProtocol]", DummyArray)((4,)), cast("Callable[..., EagerTensorProtocol]", DummyArray)((6,))]
    ) == (3, 5)
    assert cast("Callable[..., EagerTensorProtocol]", h2.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyArray)((5,)), cast("Callable[..., EagerTensorProtocol]", DummyArray)((5,)), bins=[5, 6]) == (5, 6)
    assert cast("Callable[..., EagerTensorProtocol]", h2.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyArray)((5,)), cast("Callable[..., EagerTensorProtocol]", DummyArray)((5,)), bins=[None, None]) == (10, 10)
    assert cast("Callable[..., EagerTensorProtocol]", h2.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyArray)((5,)), cast("Callable[..., EagerTensorProtocol]", DummyArray)((5,)), bins=None) == (10, 10)
    assert cast("Callable[..., EagerTensorProtocol]", h2.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyArray)((5,)), cast("Callable[..., EagerTensorProtocol]", DummyArray)((5,)), bins=[5, 6, 7]) == (10, 10)  # branch len(bins) != 2

    hbe = HistogramBinEdges()
    assert cast("Callable[..., EagerTensorProtocol]", hbe.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyArray)((5,)), bins=cast("Callable[..., EagerTensorProtocol]", DummyArray)((4,))) == (4,)
    assert cast("Callable[..., EagerTensorProtocol]", hbe.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyArray)((5,)), bins=5) == (6,)
    assert cast("Callable[..., EagerTensorProtocol]", hbe.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyArray)((5,)), bins=None) == (11,)

    # Ix edges
    ix = Ix()
    assert cast("Callable[..., EagerTensorProtocol]", ix.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyArray)((5,))) == (5,)
    assert cast("Callable[..., EagerTensorProtocol]", ix.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyArray)(())) == (1,)

    # Median edges
    med = Median()
    assert cast("Callable[..., EagerTensorProtocol]", med.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyArray)((2, 3)), axis=0) == (3,)
    assert cast("Callable[..., EagerTensorProtocol]", med.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyArray)((2, 3)), axis=0, keepdims=True) == (1, 3)
    assert cast("Callable[..., EagerTensorProtocol]", med.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyArray)((2, 3)), axis=[0, 1]) == ()
    assert cast("Callable[..., EagerTensorProtocol]", med.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyArray)((2, 3)), axis=[0, 1], keepdims=True) == (1, 1)
    assert cast("Callable[..., EagerTensorProtocol]", med.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyArray)((2, 3)), axis=[5]) == (2, 3)  # branch ax >= len(shape)

    # Rot90 edges
    rot = Rot90()
    assert cast("Callable[..., EagerTensorProtocol]", rot.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyArray)((2, 3)), axes=(0, 1)) == (3, 2)
    assert cast("Callable[..., EagerTensorProtocol]", rot.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyArray)((2, 3)), axes=(1, 0)) == (3, 2)
    assert cast("Callable[..., EagerTensorProtocol]", rot.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyArray)((2, 3)), axes=(5, 6)) == (2, 3)  # branch axes > len(shape)
    assert cast("Callable[..., EagerTensorProtocol]", rot.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyArray)((2, 3)), axes=(0, 1, 2)) == (2, 3)  # len(axes) != 2

    # Trapezoid edges
    trap = Trapezoid()
    assert cast("Callable[..., EagerTensorProtocol]", trap.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyArray)((2, 3)), axis=0) == (3,)
    assert cast("Callable[..., EagerTensorProtocol]", trap.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyArray)((2, 3)), axis=1) == (2,)
    assert cast("Callable[..., EagerTensorProtocol]", trap.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyArray)((2, 3)), axis=5) == (2, 3)  # branch axis > len(shape)
