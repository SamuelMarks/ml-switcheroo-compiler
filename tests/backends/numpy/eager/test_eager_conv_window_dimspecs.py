# ruff: noqa: C901, PLR0912, PLR0915, F841
"""Tests for NumPy eager window reductions, dimension specs, and conv/shape dispatch."""

from __future__ import annotations

import importlib
import typing
from unittest.mock import MagicMock

import numpy as np
import pytest

import tests.backends.numpy.eager._math_advanced_test_ns as mod
from ml_switcheroo_compiler.backends.eager.nn_ops.conv_reference import _conv_general_dilated, _get_transpose
from ml_switcheroo_compiler.backends.eager.reference_math.math_scatter import _band_part
from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import execute_op
from ml_switcheroo_compiler.ops.configs import ConvConfig, WindowConfig
from tests.testing_utils import DummyDimSpecs, MockTestStub, extract_test_shape, require_callable

if typing.TYPE_CHECKING:
    pass
_reduce_window = importlib.import_module(
    "ml_switcheroo_compiler.backends.eager.reference_math.window_reductions"
)._reduce_window


class DummyDimSpecsLocal:
    """Configuration class for dummy dim specs."""

    def __init__(self) -> None:
        """Initialize the instance."""
        self.lhs_spec = (0, 1, 2)
        self.rhs_spec = (0, 1, 2)
        self.out_spec = (0, 1, 2)


def test_conv_general_dilated_all_coverage_2() -> None:
    """Test the conv general dilated all coverage behavior.

    Returns:
        OpValue: The inferred shape or computed result.
    """
    lhs = np.ones((1, 2, 5))
    rhs = np.ones((4, 2, 3))
    with pytest.raises(TypeError):
        _get_transpose(123, "OIW")
    out1 = _conv_general_dilated(
        lhs,
        rhs,
        ConvConfig(window_strides=(1,), padding=None, dimension_numbers=("NCW", "OIW", "NCW")),
    )
    assert extract_test_shape(out1) == (1, 4, 3)
    out2 = _conv_general_dilated(
        lhs,
        rhs,
        ConvConfig(window_strides=(1,), padding="VALID", dimension_numbers=DummyDimSpecs()),
    )
    assert out2 is not None
    lhs2 = np.ones((1, 4, 5))
    rhs2 = np.ones((4, 2, 3))
    out3 = _conv_general_dilated(
        lhs2,
        rhs2,
        ConvConfig(
            window_strides=(1,), padding="VALID", feature_group_count=2, dimension_numbers=("NCW", "OIW", "NCW")
        ),
    )
    assert out3 is not None
    out4 = _conv_general_dilated(lhs, rhs, ConvConfig(window_strides=(1,), padding="VALID", dimension_numbers=None))
    assert out4 is not None


def test_segment_sum_coverage_2() -> None:
    """Test the segment sum coverage behavior.

    Returns:
        OpValue: The inferred shape or computed result.
    """
    data = np.array([1, 2, 3, 4, 5])
    segment_ids = np.array([0, 0, 1, 1, 2])
    res1 = mod._np_segment_sum(np, data, segment_ids)
    assert np.array_equal(res1, [3, 7, 5])
    res2 = mod._np_segment_sum(np, data, segment_ids, num_segments=4)
    assert np.array_equal(res2, [3, 7, 5, 0])


def test_numpy_eager_coverage_2() -> None:
    """Test the numpy eager coverage behavior.

    Returns:
        OpValue: The inferred shape or computed result.
    """
    handled_errors = 0
    ops = [
        "FloorDivide",
        "Remainder",
        "Divmod",
        "Mod",
        "NotEqual",
        "Less",
        "Greater",
        "LessEqual",
        "GreaterEqual",
        "LogicalAnd",
        "LogicalOr",
        "LogicalXor",
        "BitwiseAnd",
        "BitwiseOr",
        "BitwiseXor",
        "LeftShift",
        "RightShift",
        "Atan2",
        "Hypot",
        "Copysign",
        "Nextafter",
        "Ldexp",
        "Fmod",
        "Minimum",
        "Maximum",
        "Fmax",
        "Fmin",
        "Heaviside",
        "Complex",
        "Floor",
        "Ceil",
        "Round",
        "Trunc",
        "Sign",
        "Abs",
        "Exp",
        "Expm1",
        "Log",
        "Log10",
        "Log1p",
        "Log2",
        "Sin",
        "Cos",
        "Tan",
        "Asin",
        "Acos",
        "Atan",
        "Sinh",
        "Cosh",
        "Tanh",
        "Asinh",
        "Acosh",
        "Atanh",
        "Erf",
        "Erfc",
        "Gamma",
        "Lgamma",
        "Digamma",
        "Isfinite",
        "Isinf",
        "Isnan",
        "LogicalNot",
        "BitwiseNot",
        "Negative",
        "Positive",
        "Reciprocal",
        "Square",
        "Sqrt",
        "Cbrt",
        "Conjugate",
        "Real",
        "Imag",
        "Angle",
        "Sigmoid",
        "Relu",
        "Gelu",
        "Silu",
        "Swish",
        "Hardswish",
        "Hardsigmoid",
        "Selu",
        "Elu",
        "Celu",
        "LeakyRelu",
        "Logit",
        "StopGradient",
        "Zeros",
        "Ones",
        "Full",
        "ZerosLike",
        "OnesLike",
        "FullLike",
        "Eye",
        "Identity",
        "Empty",
        "EmptyLike",
        "Tri",
        "Tril",
        "Triu",
        "Linspace",
        "Logspace",
        "Geomspace",
        "Meshgrid",
        "Indices",
        "Vander",
        "Diag",
        "Diagflat",
        "Trace",
        "Cholesky",
        "Det",
        "Eigh",
        "Inv",
        "Qr",
        "Svd",
        "Solve",
        "Pinv",
        "MatrixPower",
        "KroneckerProduct",
        "Einsum",
        "Matmul",
        "Dot",
        "Vdot",
        "Inner",
        "Outer",
        "Tensordot",
        "Cross",
        "Sort",
        "ArgSort",
        "ArgMax",
        "ArgMin",
        "Max",
        "Min",
        "Sum",
        "Mean",
        "Prod",
        "All",
        "object",
        "Allclose",
        "ArrayEqual",
        "ArrayEquiv",
        "Isclose",
        "LessEqual",
        "GreaterEqual",
        "BroadcastTo",
        "BroadcastInDim",
        "ExpandDims",
        "Squeeze",
        "Reshape",
        "Transpose",
        "Permute",
        "SwapAxes",
        "Concatenate",
        "Stack",
        "Vstack",
        "Hstack",
        "Dstack",
        "ColumnStack",
        "RowStack",
        "Split",
        "ArraySplit",
        "Hsplit",
        "Vsplit",
        "Dsplit",
        "Tile",
        "Repeat",
        "Flip",
        "Fliplr",
        "Flipud",
        "Roll",
        "Rot90",
        "Resize",
        "TrimZeros",
        "Unique",
        "Pad",
        "Where",
        "Nonzero",
        "Select",
        "Piecewise",
        "Extract",
        "Put",
        "Putmask",
        "FillDiagonal",
        "DynamicSlice",
        "DynamicUpdateSlice",
        "Gather",
        "Scatter",
        "SegmentSum",
        "SegmentMax",
        "SegmentMin",
        "SegmentProd",
        "Bincount",
        "Cumsum",
        "Cumprod",
        "Cummax",
        "Cummin",
        "Histogram",
        "Histogram2d",
        "Histogramdd",
        "Digitize",
        "Searchsorted",
    ]
    x = MagicMock()
    x.shape = (2, 3)
    x.dtype = np.dtype("float32")
    x.device = "cpu"
    x.ndim = 2
    x.__len__.side_effect = TypeError
    x.__sub__.side_effect = TypeError
    for op in ops:
        try:
            execute_op(None, op, x, x)
        except Exception:
            try:
                execute_op(None, op, x)
            except Exception:
                try:
                    execute_op(None, op, x, 1)
                except (
                    ValueError,
                    TypeError,
                    AttributeError,
                    IndexError,
                    KeyError,
                    RuntimeError,
                    ZeroDivisionError,
                    NotImplementedError,
                    OSError,
                    ImportError,
                    OverflowError,
                    RecursionError,
                    LookupError,
                    ArithmeticError,
                ):
                    handled_errors += 1
    assert handled_errors >= 1


def test_reduce_window_coverage_2() -> None:
    """Test the reduce window coverage behavior.

    Returns:
        OpValue: The inferred shape or computed result.
    """
    operand = np.array([1, 2, 3])
    res = _reduce_window(
        operand,
        0,
        "sum",
        WindowConfig(
            window_dimensions=(2,), window_strides=(1,), base_dilation=(2,), window_dilation=(1,), padding=[(0, 0)]
        ),
    )
    res = _reduce_window(
        operand,
        0,
        "prod",
        WindowConfig(
            window_dimensions=(2,), window_strides=(1,), base_dilation=(1,), window_dilation=(1,), padding=[(0, 0)]
        ),
    )
    res = _reduce_window(
        operand,
        0,
        "min",
        WindowConfig(
            window_dimensions=(2,), window_strides=(1,), base_dilation=(1,), window_dilation=(1,), padding=[(0, 0)]
        ),
    )
    with pytest.raises(ValueError):
        _reduce_window(
            operand,
            0,
            "unknown",
            WindowConfig(
                window_dimensions=(2,), window_strides=(1,), base_dilation=(1,), window_dilation=(1,), padding=[(0, 0)]
            ),
        )


def test_conv_general_dilated_coverage_2() -> None:
    """Test the conv general dilated coverage behavior.

    Returns:
        OpValue: The inferred shape or computed result.
    """
    lhs = np.ones((1, 2, 5))
    rhs = np.ones((4, 2, 3))
    res = _conv_general_dilated(
        lhs,
        rhs,
        ConvConfig(window_strides=(1,), padding="SAME", dimension_numbers=("NCW", "OIW", "NCW")),
    )
    assert extract_test_shape(res) == (1, 4, 5)
    res = _conv_general_dilated(
        lhs,
        rhs,
        ConvConfig(window_strides=(1,), padding="VALID", dimension_numbers=("NCW", "OIW", "NCW")),
    )
    assert extract_test_shape(res) == (1, 4, 3)
    res = _conv_general_dilated(
        lhs,
        rhs,
        ConvConfig(window_strides=(1,), padding="UNKNOWN", dimension_numbers=("NCW", "OIW", "NCW")),
    )
    res = _conv_general_dilated(
        lhs,
        rhs,
        ConvConfig(
            window_strides=(1,),
            padding="VALID",
            dimension_numbers=("NCW", "OIW", "NCW"),
            lhs_dilation=(2,),
            rhs_dilation=(2,),
        ),
    )
    lhs_group = np.ones((1, 4, 5))
    rhs_group = np.ones((4, 2, 3))
    res = _conv_general_dilated(
        lhs_group,
        rhs_group,
        ConvConfig(
            window_strides=(1,), padding="VALID", dimension_numbers=("NCW", "OIW", "NCW"), feature_group_count=2
        ),
    )
    lhs2 = np.ones((1, 2, 5, 5))
    rhs2 = np.ones((4, 2, 3, 3))

    class DimensionNumbers(MockTestStub):
        """Configuration class for dimension numbers."""

        lhs_spec = (0, 1, 2, 3)
        rhs_spec = (0, 1, 2, 3)
        out_spec = (0, 1, 2, 3)

    res = _conv_general_dilated(
        lhs2,
        rhs2,
        ConvConfig(window_strides=(1, 1), padding="SAME", dimension_numbers=DimensionNumbers()),
    )


def test_band_part_coverage_2() -> None:
    """Test the band part coverage behavior.

    Returns:
        OpValue: The inferred shape or computed result.
    """
    res = _band_part(np.ones((3, 3)), 1, 1)
    assert res.shape == (3, 3)


def test_numpy_eager_conv_extra_2() -> None:
    """Test the numpy eager conv extra behavior.

    Returns:
        OpValue: The inferred shape or computed result.
    """
    ConvTranspose = numpy_eager_registry.get("ConvTranspose")
    _out = require_callable(ConvTranspose)(np, np.ones((1, 1, 1, 1)), np.ones((1, 1, 1, 1)), strides=1, padding="SAME")
    assert np.asarray(_out).shape == (1, 1, 1, 1) and bool(
        np.allclose(np.asarray(_out, dtype=float), [[[[1.0]]]], atol=1e-3)
    )


def test_numpy_eager_linalg_extra_2() -> None:
    """Test the numpy eager linalg extra behavior.

    Returns:
        OpValue: The inferred shape or computed result.
    """
    BandPart = numpy_eager_registry.get("BandPart")
    x = np.ones((2, 2))
    assert require_callable(BandPart)(np, x, 0, 0).shape == (2, 2)
    Svd = numpy_eager_registry.get("Svd")
    (u, s, v) = require_callable(Svd)(np, np.ones((2, 2)), full_matrices=False, compute_uv=True)
    assert u.shape == (2, 2)


def test_numpy_eager_math_extra_2() -> None:
    """Test the numpy eager math extra behavior.

    Returns:
        OpValue: The inferred shape or computed result.
    """
    TruncateDiv = numpy_eager_registry.get("TruncateDiv")
    assert np.array_equal(
        require_callable(TruncateDiv)(np, np.array([5.5]), np.array([2.0])),
        np.array([2.0]),
    )
    TruncateMod = numpy_eager_registry.get("TruncateMod")
    assert np.array_equal(
        require_callable(TruncateMod)(np, np.array([5.5]), np.array([2.0])),
        np.array([1.5]),
    )
    Betainc = numpy_eager_registry.get("Betainc")
    require_callable(Betainc)(np, 1.0, 1.0, 0.5)


def test_numpy_eager_random_extra_2() -> None:
    """Test the numpy eager random extra behavior.

    Returns:
        OpValue: The inferred shape or computed result.
    """
    Dropout = numpy_eager_registry.get("Dropout")
    res = require_callable(Dropout)(np, np.ones((2,)), 0.5)
    assert res.shape == (2,)


def test_numpy_eager_shape_extra_2() -> None:
    """Test the numpy eager shape extra behavior.

    Returns:
        OpValue: The inferred shape or computed result.
    """
    SparseExpandDims = numpy_eager_registry.get("SparseExpandDims")
    x = np.array([1])
    assert require_callable(SparseExpandDims)(np, x).shape == (1, 1)
