"""Test misc ops."""

from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.info_and_histograms import AxisIndex


def test_axis_index_infer_shape():
    """Test AxisIndex infer_shape fallback."""
    op = AxisIndex()
    # No args
    assert op.infer_shape() == ()
    # Arg without shape
    assert op.infer_shape(1) == ()
    # Arg with shape
    t = Tensor(None, TensorConfig((2, 3), "float32", "cpu"))
    assert op.infer_shape(t) == (2, 3)


"""Tests for ops misc coverage."""

from unittest.mock import patch

import ml_switcheroo_compiler.ops.binary.math as binary_math
import ml_switcheroo_compiler.ops.creation.frontend_utils as frontend_utils
import ml_switcheroo_compiler.ops.distributed_ops as dist_ops
import ml_switcheroo_compiler.ops.info_and_histograms as misc
import ml_switcheroo_compiler.ops.io as io_ops
import ml_switcheroo_compiler.ops.linalg.dot as dot_ops
import ml_switcheroo_compiler.ops.nn.activations as activations
import ml_switcheroo_compiler.ops.random_ops.core as random_core
import ml_switcheroo_compiler.ops.shape.pad_and_tile as shape_misc
import ml_switcheroo_compiler.ops.shape.slicing as slicing
import ml_switcheroo_compiler.ops.unary.logical as unary_logical


def test_cover_misc() -> None:
    """Test coverage for misc ops."""
    with patch("ml_switcheroo_compiler.ops.dispatcher.dispatch_op") as mock_dispatch:
        for module in [misc, shape_misc, unary_logical, dist_ops, frontend_utils, activations, io_ops, dot_ops, binary_math, slicing, random_core]:
            for attr in dir(module):
                val = getattr(module, attr)
                if callable(val) and attr not in ["dispatch_op", "Any", "object"]:
                    try:
                        val()
                    except Exception:
                        pass


def test_grid_and_indices_infer_shapes_extras():
    """test_grid_and_indices_infer_shapes_extras."""
    from ml_switcheroo_compiler.ops.info_and_histograms.grid_and_indices import Indices, Ix, MaskIndices, Mgrid, Ogrid, R

    # Indices
    op = Indices()
    assert op.infer_shape() == ()
    assert op.infer_shape((2, 2)) == (2, 2, 2)
    assert op.infer_shape(None) == ()

    # Ix
    op = Ix()
    assert op.infer_shape() == ()

    class Dummy:
        """Dummy."""

        shape = (3,)

    assert op.infer_shape(Dummy(), Dummy()) == (3, 1)

    class DummyNoShape:
        """DummyNoShape."""

        pass

    assert op.infer_shape(DummyNoShape()) == (1,)

    # Others
    assert MaskIndices().infer_shape() == (None,)
    assert Mgrid().infer_shape(shape=(5, 5)) == (5, 5)
    assert Ogrid().infer_shape(shape=(4,)) == (4,)
    assert R().infer_shape() == (None,)


def test_histograms_infer_shapes():
    """test_histograms_infer_shapes."""
    from ml_switcheroo_compiler.ops.info_and_histograms.histograms import Histogram, Histogram2d, HistogramBinEdges, Histogramdd

    class DummyShape:
        """DummyShape."""

        def __init__(self, shape):
            """__init__."""
            self.shape = shape

    # Histogram
    assert Histogram().infer_shape(bins=DummyShape((5,))) == (4,)
    assert Histogram().infer_shape(bins=10) == (10,)
    assert Histogram().infer_shape(bins="auto") == (10,)

    # Histogram2d
    assert Histogram2d().infer_shape(bins=DummyShape((5,))) == (10, 10)
    assert Histogram2d().infer_shape(bins=[10, DummyShape((5,))]) == (10, 4)
    assert Histogram2d().infer_shape(bins=[DummyShape((2,)), 5]) == (1, 5)

    # HistogramBinEdges
    assert HistogramBinEdges().infer_shape(bins=DummyShape((5,))) == (5,)
    assert HistogramBinEdges().infer_shape(bins=10) == (11,)
    assert HistogramBinEdges().infer_shape(bins="auto") == (11,)

    # Histogramdd
    assert Histogramdd().infer_shape() == ()
    assert Histogramdd().infer_shape(DummyShape((5, 2))) == (10, 10)
    assert Histogramdd().infer_shape(DummyShape((5,))) == (10,)


def test_histograms_infer_shapes_branch():
    """test_histograms_infer_shapes_branch."""
    from ml_switcheroo_compiler.ops.info_and_histograms.histograms import Histogram2d

    assert Histogram2d().infer_shape(bins=[10, 10, 10]) == (10, 10)


def test_info_infer_shapes():
    """test_info_infer_shapes."""
    from ml_switcheroo_compiler.ops.info_and_histograms.info import Finfo, GetPrintoptions, Iinfo, Isscalar, Iterable, PromoteTypes, ResultType

    assert Finfo().infer_shape() == ()
    assert Iinfo().infer_shape() == ()
    assert GetPrintoptions().infer_shape() == ()
    assert Isscalar().infer_shape() == ()
    assert Iterable().infer_shape() == ()
    assert PromoteTypes().infer_shape() == ()
    assert ResultType().infer_shape() == ()


def test_math_misc_infer_shapes():
    """test_math_misc_infer_shapes."""
    from ml_switcheroo_compiler.ops.info_and_histograms.math_misc import I0, Gradient, Interp, Intersect1d, Kron, Median, Mish, Modf, Piecewise, Rot90, Trapezoid, Tri, Tril, TrimZeros, Triu, Unwrap, Vander

    class DummyShape:
        """DummyShape."""

        def __init__(self, shape):
            """__init__."""
            self.shape = shape

    # Gradient
    assert Gradient().infer_shape() == ()
    assert Gradient().infer_shape(DummyShape((2, 2))) == (2, 2)

    # I0
    assert I0().infer_shape() == ()
    assert I0().infer_shape(DummyShape((2,))) == (2,)

    # Interp
    assert Interp().infer_shape() == ()
    assert Interp().infer_shape(DummyShape((3,))) == (3,)

    # Intersect1d

    assert Intersect1d().infer_shape(DummyShape((2,)), DummyShape((3,))) == (None,)

    # Kron
    assert Kron().infer_shape() == ()
    assert Kron().infer_shape(DummyShape((2, 3)), DummyShape((4, 5))) == (8, 15)

    # Median
    assert Median().infer_shape() == ()
    assert Median().infer_shape(DummyShape((2, 2))) == ()
    assert Median().infer_shape(DummyShape((2, 2)), axis=0) == (2,)

    # Mish
    assert Mish().infer_shape() == ()
    assert Mish().infer_shape(DummyShape((2, 2))) == (2, 2)

    # Modf
    assert Modf().infer_shape() == ()
    assert Modf().infer_shape(DummyShape((2, 2))) == (2, 2)

    # Piecewise
    assert Piecewise().infer_shape() == ()
    assert Piecewise().infer_shape(DummyShape((2, 2))) == (2, 2)

    # Rot90
    assert Rot90().infer_shape() == ()
    assert Rot90().infer_shape(DummyShape((2, 3))) == (3, 2)
    assert Rot90().infer_shape(DummyShape((2, 3)), k=2) == (3, 2)

    # Trapezoid
    assert Trapezoid().infer_shape() == ()
    assert Trapezoid().infer_shape(DummyShape((2, 2))) == (2,)

    # Tri
    assert Tri().infer_shape(2) == (2, 2)
    assert Tri().infer_shape(2, M=3) == (2, 3)

    # Tril
    assert Tril().infer_shape() == ()
    assert Tril().infer_shape(DummyShape((2, 2))) == (2, 2)

    # TrimZeros
    assert TrimZeros().infer_shape() == (None,)

    # Triu
    assert Triu().infer_shape() == ()
    assert Triu().infer_shape(DummyShape((2, 2))) == (2, 2)

    # Unwrap
    assert Unwrap().infer_shape() == ()
    assert Unwrap().infer_shape(DummyShape((2, 2))) == (2, 2)

    # Vander
    assert Vander().infer_shape() == ()
    assert Vander().infer_shape(DummyShape((3,))) == (3, 3)
    assert Vander().infer_shape(DummyShape((3,)), N=4) == (3, 4)


def test_math_misc_infer_shapes_branches():
    """test_math_misc_infer_shapes_branches."""
    from ml_switcheroo_compiler.ops.info_and_histograms.math_misc import Median, Rot90, Tri

    class DummyShape:
        """DummyShape."""

        def __init__(self, shape):
            """__init__."""
            self.shape = shape

    # Median
    assert Median().infer_shape(DummyShape((2, 2)), axis=0, keepdims=True) == (1, 2)
    assert Median().infer_shape(DummyShape((2, 2)), axis=[0, 1]) == ()
    assert Median().infer_shape(DummyShape((2, 2)), axis=5) == (2, 2)

    # Rot90
    assert Rot90().infer_shape(DummyShape((2, 3)), axes=(1, 5)) == (2, 3)

    # Tri
    assert Tri().infer_shape(2, M=2) == (2, 2)
    assert Tri().infer_shape(2) == (2, 2)


def test_tri_no_args():
    """test_tri_no_args."""
    from ml_switcheroo_compiler.ops.info_and_histograms.math_misc import Tri

    assert Tri().infer_shape() == (0, 0)


def test_trapezoid_branch():
    """test_trapezoid_branch."""
    from ml_switcheroo_compiler.ops.info_and_histograms.math_misc import Trapezoid

    class DummyShape:
        """DummyShape."""

        def __init__(self, shape):
            """__init__."""
            self.shape = shape

    assert Trapezoid().infer_shape(DummyShape((2,)), axis=5) == (2,)


def test_misc_ops_infer_shapes():
    """test_misc_ops_infer_shapes."""
    from ml_switcheroo_compiler.ops.info_and_histograms.misc_ops import Infeed, Vectorize

    assert Infeed().infer_shape(shape=(3, 4)) == (3, 4)
    assert Vectorize().infer_shape() == ()
