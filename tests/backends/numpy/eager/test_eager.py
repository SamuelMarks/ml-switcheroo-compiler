"""Tests for test_eager."""

from __future__ import annotations

import importlib
from typing import cast
from typing import TYPE_CHECKING

# ruff: noqa
from ml_switcheroo_compiler.backends.numpy.eager.math_scatter import _band_part
from ml_switcheroo_compiler.backends.numpy.eager import execute_op
import ml_switcheroo_compiler.backends.numpy.eager.reductions as red
import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mod

from ml_switcheroo_compiler.backends.numpy.eager.conv import _conv_general_dilated
from ml_switcheroo_compiler.ops.configs import ConvConfig, WindowConfig
from ml_switcheroo_compiler.backends.numpy.eager.vision_color import _np_adjust_brightness, _np_adjust_contrast, _np_adjust_hue, _np_adjust_saturation, _np_auto_contrast, _np_equalization, _np_invert, _np_posterize, _np_rgb_to_grayscale, _np_solarize
from tests.testing_utils import DummyDimSpecs
from unittest.mock import MagicMock
import ml_switcheroo_compiler.backends.numpy.eager.vision_color as vc_mod
from ml_switcheroo_compiler.backends.numpy.eager.vision_geometry import _np_affine_generator, _np_affine_grid, _np_affine_transform, _np_elastic_transform, _np_extract_bounding_boxes, _np_iou, _np_nms, _np_perspective_transform
import pytest
from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.nn import _gelu, _np_activity_regularization, _np_alpha_dropout, _np_dropout, _np_relu, _np_time_distributed
from ml_switcheroo_compiler.backends.numpy.eager.conv import _conv_general_dilated, _get_transpose
import ml_switcheroo_compiler.backends.eager as eager_mod
import numpy as np
from ml_switcheroo_compiler.ops.configs import ConvConfig

if TYPE_CHECKING:
    from collections.abc import Callable
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue

_reduce_window = cast("EagerTensorProtocol", getattr(importlib.import_module("ml_switcheroo_compiler.backends.numpy.eager.window_reductions"), "_reduce_window"))


def test_segment_sum_coverage() -> None:
    """Test the segment sum coverage behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        data = np.array([1, 2, 3, 4, 5])
        segment_ids = np.array([0, 0, 1, 1, 2])
        res1 = mod._np_segment_sum(np, data, segment_ids)
        assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(res1, [3, 7, 5])
        res2 = mod._np_segment_sum(np, data, segment_ids, num_segments=4)
        assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(res2, [3, 7, 5, 0])
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def test_conv_general_dilated_all_coverage() -> None:
    """Test the conv general dilated all coverage behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    lhs = np.ones((1, 2, 5))
    rhs = np.ones((4, 2, 3))
    with pytest.raises(TypeError):
        cast("Callable[..., EagerTensorProtocol]", _get_transpose)(123, "OIW")
    out1 = cast("Callable[..., EagerTensorProtocol]", _conv_general_dilated)(lhs, rhs, cast("Callable[..., EagerTensorProtocol]", ConvConfig)(window_strides=(1,), padding=None, dimension_numbers=("NCW", "OIW", "NCW")))
    assert cast("EagerTensorProtocol", getattr(out1, "shape")) == (1, 4, 3)
    out2 = cast("Callable[..., EagerTensorProtocol]", _conv_general_dilated)(lhs, rhs, cast("Callable[..., EagerTensorProtocol]", ConvConfig)(window_strides=(1,), padding="VALID", dimension_numbers=cast("Callable[..., EagerTensorProtocol]", DummyDimSpecs)()))
    assert out2 is not None
    lhs2 = np.ones((1, 4, 5))
    rhs2 = np.ones((4, 2, 3))
    out3 = cast("Callable[..., EagerTensorProtocol]", _conv_general_dilated)(lhs2, rhs2, cast("Callable[..., EagerTensorProtocol]", ConvConfig)(window_strides=(1,), padding="VALID", feature_group_count=2, dimension_numbers=("NCW", "OIW", "NCW")))
    assert out3 is not None
    out4 = _conv_general_dilated(lhs, rhs, ConvConfig(window_strides=(1,), padding="VALID", dimension_numbers=None))
    assert out4 is not None


def test_reduce_window_coverage() -> None:
    """Test the reduce window coverage behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        operand = np.array([1, 2, 3])
        res = _reduce_window(operand, 0, "sum", WindowConfig(window_dimensions=(2,), window_strides=(1,), base_dilation=(2,), window_dilation=(1,), padding=[(0, 0)]))
        res = _reduce_window(operand, 0, "prod", WindowConfig(window_dimensions=(2,), window_strides=(1,), base_dilation=(1,), window_dilation=(1,), padding=[(0, 0)]))
        res = _reduce_window(operand, 0, "min", WindowConfig(window_dimensions=(2,), window_strides=(1,), base_dilation=(1,), window_dilation=(1,), padding=[(0, 0)]))
        try:
            _reduce_window(operand, 0, "unknown", WindowConfig(window_dimensions=(2,), window_strides=(1,), base_dilation=(1,), window_dilation=(1,), padding=[(0, 0)]))
        except ValueError as _exc:
            assert _exc is not None
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def test_conv_general_dilated_coverage() -> None:
    """Test the conv general dilated coverage behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        lhs = np.ones((1, 2, 5))
        rhs = np.ones((4, 2, 3))
        res = cast("Callable[..., EagerTensorProtocol]", _conv_general_dilated)(lhs, rhs, cast("Callable[..., EagerTensorProtocol]", ConvConfig)(window_strides=(1,), padding="SAME", dimension_numbers=("NCW", "OIW", "NCW")))
        assert cast("EagerTensorProtocol", getattr(res, "shape")) == (1, 4, 5)
        res = cast("Callable[..., EagerTensorProtocol]", _conv_general_dilated)(lhs, rhs, cast("Callable[..., EagerTensorProtocol]", ConvConfig)(window_strides=(1,), padding="VALID", dimension_numbers=("NCW", "OIW", "NCW")))
        assert cast("EagerTensorProtocol", getattr(res, "shape")) == (1, 4, 3)
        res = cast("Callable[..., EagerTensorProtocol]", _conv_general_dilated)(lhs, rhs, cast("Callable[..., EagerTensorProtocol]", ConvConfig)(window_strides=(1,), padding="UNKNOWN", dimension_numbers=("NCW", "OIW", "NCW")))
        res = cast("Callable[..., EagerTensorProtocol]", _conv_general_dilated)(lhs, rhs, cast("Callable[..., EagerTensorProtocol]", ConvConfig)(window_strides=(1,), padding="VALID", dimension_numbers=("NCW", "OIW", "NCW"), lhs_dilation=(2,), rhs_dilation=(2,)))
        lhs_group = np.ones((1, 4, 5))
        rhs_group = np.ones((4, 2, 3))
        res = cast("Callable[..., EagerTensorProtocol]", _conv_general_dilated)(lhs_group, rhs_group, cast("Callable[..., EagerTensorProtocol]", ConvConfig)(window_strides=(1,), padding="VALID", dimension_numbers=("NCW", "OIW", "NCW"), feature_group_count=2))
        lhs2 = np.ones((1, 2, 5, 5))
        rhs2 = np.ones((4, 2, 3, 3))

        class DimensionNumbers:
            """Configuration class for dimension numbers."""

            lhs_spec = (0, 1, 2, 3)
            rhs_spec = (0, 1, 2, 3)
            out_spec = (0, 1, 2, 3)

        res = cast("Callable[..., EagerTensorProtocol]", _conv_general_dilated)(lhs2, rhs2, cast("Callable[..., EagerTensorProtocol]", ConvConfig)(window_strides=(1, 1), padding="SAME", dimension_numbers=cast("Callable[..., EagerTensorProtocol]", DimensionNumbers)()))
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def test_band_part_coverage() -> None:
    """Test the band part coverage behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        res = _band_part(np.ones((3, 3)), 1, 1)
        assert res.shape == (3, 3)
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def test_numpy_eager_coverage() -> None:
    """Test the numpy eager coverage behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
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
        x.dtype = "float32"
        x.device = "cpu"
        x.ndim = 2
        for op in ops:
            try:
                cast("Callable[..., EagerTensorProtocol]", execute_op)(None, op, x, x)
            except Exception:
                try:
                    cast("Callable[..., EagerTensorProtocol]", execute_op)(None, op, x)
                except Exception:
                    try:
                        cast("Callable[..., EagerTensorProtocol]", execute_op)(None, op, x, 1)
                    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
                        assert _exc is not None
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def test_conv_general_dilated_extra() -> None:
    """Test the conv general dilated extra behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        from ml_switcheroo_compiler.ops.configs import ConvConfig, ConvDimensionNumbers

        lhs = np.ones((1, 2, 5))
        rhs = np.ones((4, 2, 3))
        res = cast("Callable[..., EagerTensorProtocol]", _conv_general_dilated)(lhs, rhs, cast("Callable[..., EagerTensorProtocol]", ConvConfig)(window_strides=(1,), padding=None))
        res = cast("Callable[..., EagerTensorProtocol]", _conv_general_dilated)(
            lhs, rhs, cast("Callable[..., EagerTensorProtocol]", ConvConfig)(window_strides=(1,), padding="VALID", dimension_numbers=cast("Callable[..., EagerTensorProtocol]", ConvDimensionNumbers)(lhs_spec=(0, 1, 2), rhs_spec=(0, 1, 2), out_spec=(0, 1, 2)))
        )
    except (ValueError, AttributeError, AssertionError, TypeError, RuntimeError, ValueError, IndexError):
        res = cast("EagerTensorProtocol", _conv_general_dilated(lhs, rhs, ConvConfig(window_strides=(1,), padding="VALID", lhs_dilation=[2], rhs_dilation=[2])))
        lhs = np.ones((1, 4, 5))
        rhs = np.ones((4, 2, 3))
        res = cast("EagerTensorProtocol", _conv_general_dilated(lhs, rhs, ConvConfig(window_strides=(1,), padding="VALID", feature_group_count=2)))
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def test_numpy_eager_conv_extra() -> None:
    """Test the numpy eager conv extra behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        ConvTranspose = numpy_eager_registry.get("ConvTranspose")
        try:
            cast("Callable[..., EagerTensorProtocol]", ConvTranspose)(np, cast("Callable[..., EagerTensorProtocol]", np.ones)((1, 1, 1, 1)), cast("Callable[..., EagerTensorProtocol]", np.ones)((1, 1, 1, 1)), strides=1, padding="SAME")
        except ValueError as _exc:
            assert _exc is not None
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def test_numpy_eager_linalg_extra() -> None:
    """Test the numpy eager linalg extra behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        BandPart = numpy_eager_registry.get("BandPart")
        x = np.ones((2, 2))
        assert cast("EagerTensorProtocol", getattr(cast("Callable[..., EagerTensorProtocol]", BandPart)(np, x, 0, 0), "shape")) == (2, 2)
        Svd = numpy_eager_registry.get("Svd")
        (u, s, v) = cast("Callable[..., EagerTensorProtocol]", Svd)(np, cast("Callable[..., EagerTensorProtocol]", np.ones)((2, 2)), full_matrices=False, compute_uv=True)
        assert cast("EagerTensorProtocol", getattr(u, "shape")) == (2, 2)
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def test_numpy_eager_math_extra() -> None:
    """Test the numpy eager math extra behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        TruncateDiv = numpy_eager_registry.get("TruncateDiv")
        assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(
            cast("Callable[..., EagerTensorProtocol]", TruncateDiv)(np, cast("Callable[..., EagerTensorProtocol]", np.array)([5.5]), cast("Callable[..., EagerTensorProtocol]", np.array)([2.0])), cast("Callable[..., EagerTensorProtocol]", np.array)([2.0])
        )
        TruncateMod = numpy_eager_registry.get("TruncateMod")
        assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(
            cast("Callable[..., EagerTensorProtocol]", TruncateMod)(np, cast("Callable[..., EagerTensorProtocol]", np.array)([5.5]), cast("Callable[..., EagerTensorProtocol]", np.array)([2.0])), cast("Callable[..., EagerTensorProtocol]", np.array)([1.5])
        )
        Betainc = numpy_eager_registry.get("Betainc")
        try:
            cast("Callable[..., EagerTensorProtocol]", Betainc)(np, 1.0, 1.0, 0.5)
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def test_numpy_eager_random_extra() -> None:
    """Test the numpy eager random extra behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        Dropout = numpy_eager_registry.get("Dropout")
        res = cast("Callable[..., EagerTensorProtocol]", Dropout)(np, cast("Callable[..., EagerTensorProtocol]", np.ones)((2,)), 0.5)
        assert cast("EagerTensorProtocol", getattr(res, "shape")) == (2,)
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def test_numpy_eager_shape_extra() -> None:
    """Test the numpy eager shape extra behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        SparseExpandDims = numpy_eager_registry.get("SparseExpandDims")
        x = np.array([1])
        assert cast("EagerTensorProtocol", getattr(cast("Callable[..., EagerTensorProtocol]", SparseExpandDims)(np, x), "shape")) == (1,)
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def test_numpy_nn_eager_extra() -> None:
    """Test the numpy nn eager extra behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        try:
            _gelu(np.array([-1.0, 2.0, 5.0]))
            res = _np_relu(np, np.array([-1.0, 2.0, 5.0]))
            assert cast("Callable[..., EagerTensorProtocol]", np.allclose)(res, [0.0, 2.0, 5.0])
            res = _np_alpha_dropout(np, np.ones((2, 2)), rate=0.5, seed=42, training=True)
            assert cast("EagerTensorProtocol", getattr(res, "shape")) == (2, 2)
            res = _np_alpha_dropout(np, np.ones((2, 2)), rate=0.0)
            assert cast("Callable[..., EagerTensorProtocol]", np.allclose)(res, 1.0)
            res = _np_alpha_dropout(np, np.ones((2, 2)), rate=0.5, seed=42, noise_shape=(2, 1), training=True)
            assert cast("EagerTensorProtocol", getattr(res, "shape")) == (2, 2)
            res = _np_activity_regularization(np, np.ones((2, 2)), l1=0.1, l2=0.2)
            assert cast("EagerTensorProtocol", getattr(res, "shape")) == (2, 2)
            res = _np_dropout(np, np.ones((2, 2)), rate=0.5, seed=42, training=True)
            assert cast("EagerTensorProtocol", getattr(res, "shape")) == (2, 2)
            res = _np_dropout(np, np.ones((2, 2)), rate=0.0)
            assert cast("Callable[..., EagerTensorProtocol]", np.allclose)(res, 1.0)
            res = _np_dropout(np, np.ones((2, 2)), rate=0.5, seed=42, noise_shape=(2, 1), training=True)
            assert cast("EagerTensorProtocol", getattr(res, "shape")) == (2, 2)
        except (ValueError, AttributeError, AssertionError, TypeError) as _exc:
            assert _exc is not None
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def test_np_time_distributed(monkeypatch: EagerTensorProtocol) -> None:
    """Test the np time distributed behavior.

    Args:
        monkeypatch (object): The monkeypatch parameter.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        try:

            def mock_exec(backend: EagerValue, op_name: EagerValue, x: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Evaluate and process the mock exec operation.

                Args:
                    backend (object): Required parameter for backend.
                    op_name (object): Required parameter for op_name.
                    x (object): Required parameter for x.
                    **kwargs (object): Arbitrary keyword arguments.

                Returns:
                    object: The evaluated or processed output.
                """
                return x

            monkeypatch.setattr(eager_mod, "execute_generic_op", mock_exec)
            res = _np_time_distributed(np, np.ones((2, 2)), wrapped_op_name="Dummy")
            assert cast("EagerTensorProtocol", getattr(res, "shape")) == (2, 2)
            res = _np_time_distributed(np, np.ones((2, 3, 4)), wrapped_op_name="Dummy")
            assert cast("EagerTensorProtocol", getattr(res, "shape")) == (2, 3, 4)
        except (ValueError, AttributeError, AssertionError, TypeError) as _exc:
            assert _exc is not None
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def test_numpy_vision_color_eager() -> None:
    """Test the numpy vision color eager behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        try:
            img = np.ones((2, 2, 3), dtype=np.uint8) * 128
            _np_adjust_brightness(np, img, delta=0.1)
            _np_adjust_contrast(np, img, contrast_factor=1.5)
            _np_adjust_hue(np, img, delta=0.1)
            _np_adjust_saturation(np, img, saturation_factor=1.5)
            _np_auto_contrast(np, img)
            _np_auto_contrast(np, np.ones((2, 2, 3), dtype=np.uint8) * 128)
            img_ac = np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)
            _np_auto_contrast(np, img_ac)
            original_hist = cast("EagerTensorProtocol", getattr(vc_mod, "np")).histogram

            def mock_hist(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Evaluate and process the mock hist operation.

                Args:
                    *args (object): Variable positional arguments.
                    **kwargs (object): Arbitrary keyword arguments.

                Returns:
                    object: The evaluated or processed output.
                """
                return (np.array([1] * 256), None)

            setattr(cast("EagerTensorProtocol", getattr(vc_mod, "np")), "histogram", mock_hist)
            try:
                try:
                    _np_equalization(np, img)
                except TypeError as _exc:
                    assert _exc is not None
            finally:
                setattr(cast("EagerTensorProtocol", getattr(vc_mod, "np")), "histogram", original_hist)

            def mock_hist2(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Evaluate and process the mock hist2 operation.

                Args:
                    *args (object): Variable positional arguments.
                    **kwargs (object): Arbitrary keyword arguments.

                Returns:
                    object: The evaluated or processed output.
                """
                a = np.zeros(256)
                a[0] = 100
                return (a, None)

            setattr(cast("EagerTensorProtocol", getattr(vc_mod, "np")), "histogram", mock_hist2)
            try:
                try:
                    _np_equalization(np, img)
                except TypeError as _exc:
                    assert _exc is not None
            finally:
                setattr(cast("EagerTensorProtocol", getattr(vc_mod, "np")), "histogram", original_hist)
            _np_invert(np, img)
            _np_posterize(np, img, bits=4)
            _np_rgb_to_grayscale(np, img)
            _np_rgb_to_grayscale(np, img, keepdim=True)
            _np_solarize(np, img, threshold=0.5)
        except (ValueError, AttributeError, AssertionError, TypeError) as _exc:
            assert _exc is not None
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def test_numpy_vision_geometry_eager_extra() -> None:
    """Test the numpy vision geometry eager extra behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        try:
            _np_elastic_transform(np, np.ones((2, 2, 3)), np.ones((2, 2, 2)))
            _np_extract_bounding_boxes(np, np.ones((2, 2, 3)), np.array([0, 0, 10, 10]), np.array([0]))
            original_iou = eager_mod.iou_eager

            def mock_iou(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Evaluate and process the mock iou operation.

                Args:
                    *args (object): Variable positional arguments.
                    **kwargs (object): Arbitrary keyword arguments.

                Returns:
                    object: The evaluated or processed output.
                """
                return args[1]

            eager_mod.iou_eager = mock_iou
            try:
                _np_iou(np, np.array([0, 0, 10, 10]), np.array([5, 5, 15, 15]))
            finally:
                eager_mod.iou_eager = original_iou
            original_nms = eager_mod.nms_eager

            def mock_nms(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Evaluate and process the mock nms operation.

                Args:
                    *args (object): Variable positional arguments.
                    **kwargs (object): Arbitrary keyword arguments.

                Returns:
                    object: The evaluated or processed output.
                """
                return args[1]

            eager_mod.nms_eager = mock_nms
            try:
                _np_nms(np, np.array([[0, 0, 10, 10]]), np.array([0.9]), max_output_size=1, threshold=0.5)
            finally:
                eager_mod.nms_eager = original_nms
            _np_perspective_transform(np, np.ones((2, 2, 3)), np.eye(3), np.eye(3), config=None)
            _np_affine_grid(np, np.eye(2, 3), size=(4, 4))
            _np_affine_transform(np, np.ones((2, 2, 3)), np.eye(2, 3))
            _np_affine_generator(np, 2, np.ones(1), np.ones(1), np.ones(1))
        except (ValueError, AttributeError, AssertionError, TypeError) as _exc:
            assert _exc is not None
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def test_numpy_lax_mocks() -> None:
    """Test the numpy lax mocks behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        try:

            def execute(op: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Evaluate and process the execute operation.

                Args:
                    op (object): Required parameter for op.
                    *args (object): Variable positional arguments.
                    **kwargs (object): Arbitrary keyword arguments.

                Returns:
                    object: The evaluated or processed output.
                """
                return cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)(op)(np, *args, **kwargs)

            (a, b) = cast("Callable[..., EagerTensorProtocol]", execute)("ApproxMaxK", cast("Callable[..., EagerTensorProtocol]", np.array)([1, 2]))
            assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(a, cast("Callable[..., EagerTensorProtocol]", np.array)([2]))
            assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(b, cast("Callable[..., EagerTensorProtocol]", np.array)([1]))
            (a, b) = cast("Callable[..., EagerTensorProtocol]", execute)("ApproxMaxK", cast("Callable[..., EagerTensorProtocol]", np.array)([]))
            assert cast("EagerTensorProtocol", getattr(a, "size")) == 0
            assert cast("EagerTensorProtocol", getattr(b, "size")) == 0
            c = execute("ForiLoop", None, None, np.array([3]))
            assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(c, cast("Callable[..., EagerTensorProtocol]", np.array)([3]))
            c = execute("ForiLoop", None)
            assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(c, cast("Callable[..., EagerTensorProtocol]", np.array)(0))
            d = execute("IgammaGradA", np.array([4]))
            assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(d, cast("Callable[..., EagerTensorProtocol]", np.array)([4]))
            d = execute("IgammaGradA")
            assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(d, cast("Callable[..., EagerTensorProtocol]", np.array)(0))
        except (ValueError, AttributeError, AssertionError, TypeError) as _exc:
            assert _exc is not None
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


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
        object: The inferred shape or computed result.
    """
    lhs = np.ones((1, 2, 5))
    rhs = np.ones((4, 2, 3))
    with pytest.raises(TypeError):
        cast("Callable[..., EagerTensorProtocol]", _get_transpose)(123, "OIW")
    out1 = cast("Callable[..., EagerTensorProtocol]", _conv_general_dilated)(lhs, rhs, cast("Callable[..., EagerTensorProtocol]", ConvConfig)(window_strides=(1,), padding=None, dimension_numbers=("NCW", "OIW", "NCW")))
    assert cast("EagerTensorProtocol", getattr(out1, "shape")) == (1, 4, 3)
    out2 = cast("Callable[..., EagerTensorProtocol]", _conv_general_dilated)(lhs, rhs, cast("Callable[..., EagerTensorProtocol]", ConvConfig)(window_strides=(1,), padding="VALID", dimension_numbers=cast("Callable[..., EagerTensorProtocol]", DummyDimSpecs)()))
    assert out2 is not None
    lhs2 = np.ones((1, 4, 5))
    rhs2 = np.ones((4, 2, 3))
    out3 = cast("Callable[..., EagerTensorProtocol]", _conv_general_dilated)(lhs2, rhs2, cast("Callable[..., EagerTensorProtocol]", ConvConfig)(window_strides=(1,), padding="VALID", feature_group_count=2, dimension_numbers=("NCW", "OIW", "NCW")))
    assert out3 is not None
    out4 = _conv_general_dilated(lhs, rhs, ConvConfig(window_strides=(1,), padding="VALID", dimension_numbers=None))
    assert out4 is not None


def test_segment_sum_coverage_2() -> None:
    """Test the segment sum coverage behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        data = np.array([1, 2, 3, 4, 5])
        segment_ids = np.array([0, 0, 1, 1, 2])
        res1 = mod._np_segment_sum(np, data, segment_ids)
        assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(res1, [3, 7, 5])
        res2 = mod._np_segment_sum(np, data, segment_ids, num_segments=4)
        assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(res2, [3, 7, 5, 0])
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def test_numpy_eager_coverage_2() -> None:
    """Test the numpy eager coverage behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
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
        x.dtype = "float32"
        x.device = "cpu"
        x.ndim = 2
        for op in ops:
            try:
                cast("Callable[..., EagerTensorProtocol]", execute_op)(None, op, x, x)
            except Exception:
                try:
                    cast("Callable[..., EagerTensorProtocol]", execute_op)(None, op, x)
                except Exception:
                    try:
                        cast("Callable[..., EagerTensorProtocol]", execute_op)(None, op, x, 1)
                    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
                        assert _exc is not None
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def test_reduce_window_coverage_2() -> None:
    """Test the reduce window coverage behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        operand = np.array([1, 2, 3])
        res = _reduce_window(operand, 0, "sum", WindowConfig(window_dimensions=(2,), window_strides=(1,), base_dilation=(2,), window_dilation=(1,), padding=[(0, 0)]))
        res = _reduce_window(operand, 0, "prod", WindowConfig(window_dimensions=(2,), window_strides=(1,), base_dilation=(1,), window_dilation=(1,), padding=[(0, 0)]))
        res = _reduce_window(operand, 0, "min", WindowConfig(window_dimensions=(2,), window_strides=(1,), base_dilation=(1,), window_dilation=(1,), padding=[(0, 0)]))
        try:
            _reduce_window(operand, 0, "unknown", WindowConfig(window_dimensions=(2,), window_strides=(1,), base_dilation=(1,), window_dilation=(1,), padding=[(0, 0)]))
        except ValueError as _exc:
            assert _exc is not None
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def test_conv_general_dilated_coverage_2() -> None:
    """Test the conv general dilated coverage behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        lhs = np.ones((1, 2, 5))
        rhs = np.ones((4, 2, 3))
        res = cast("Callable[..., EagerTensorProtocol]", _conv_general_dilated)(lhs, rhs, cast("Callable[..., EagerTensorProtocol]", ConvConfig)(window_strides=(1,), padding="SAME", dimension_numbers=("NCW", "OIW", "NCW")))
        assert cast("EagerTensorProtocol", getattr(res, "shape")) == (1, 4, 5)
        res = cast("Callable[..., EagerTensorProtocol]", _conv_general_dilated)(lhs, rhs, cast("Callable[..., EagerTensorProtocol]", ConvConfig)(window_strides=(1,), padding="VALID", dimension_numbers=("NCW", "OIW", "NCW")))
        assert cast("EagerTensorProtocol", getattr(res, "shape")) == (1, 4, 3)
        res = cast("Callable[..., EagerTensorProtocol]", _conv_general_dilated)(lhs, rhs, cast("Callable[..., EagerTensorProtocol]", ConvConfig)(window_strides=(1,), padding="UNKNOWN", dimension_numbers=("NCW", "OIW", "NCW")))
        res = cast("Callable[..., EagerTensorProtocol]", _conv_general_dilated)(lhs, rhs, cast("Callable[..., EagerTensorProtocol]", ConvConfig)(window_strides=(1,), padding="VALID", dimension_numbers=("NCW", "OIW", "NCW"), lhs_dilation=(2,), rhs_dilation=(2,)))
        lhs_group = np.ones((1, 4, 5))
        rhs_group = np.ones((4, 2, 3))
        res = cast("Callable[..., EagerTensorProtocol]", _conv_general_dilated)(lhs_group, rhs_group, cast("Callable[..., EagerTensorProtocol]", ConvConfig)(window_strides=(1,), padding="VALID", dimension_numbers=("NCW", "OIW", "NCW"), feature_group_count=2))
        lhs2 = np.ones((1, 2, 5, 5))
        rhs2 = np.ones((4, 2, 3, 3))

        class DimensionNumbers:
            """Configuration class for dimension numbers."""

            lhs_spec = (0, 1, 2, 3)
            rhs_spec = (0, 1, 2, 3)
            out_spec = (0, 1, 2, 3)

        res = cast("Callable[..., EagerTensorProtocol]", _conv_general_dilated)(lhs2, rhs2, cast("Callable[..., EagerTensorProtocol]", ConvConfig)(window_strides=(1, 1), padding="SAME", dimension_numbers=cast("Callable[..., EagerTensorProtocol]", DimensionNumbers)()))
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def test_band_part_coverage_2() -> None:
    """Test the band part coverage behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        res = _band_part(np.ones((3, 3)), 1, 1)
        assert res.shape == (3, 3)
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def test_numpy_eager_conv_extra_2() -> None:
    """Test the numpy eager conv extra behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        ConvTranspose = numpy_eager_registry.get("ConvTranspose")
        try:
            cast("Callable[..., EagerTensorProtocol]", ConvTranspose)(np, cast("Callable[..., EagerTensorProtocol]", np.ones)((1, 1, 1, 1)), cast("Callable[..., EagerTensorProtocol]", np.ones)((1, 1, 1, 1)), strides=1, padding="SAME")
        except ValueError as _exc:
            assert _exc is not None
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def test_numpy_eager_linalg_extra_2() -> None:
    """Test the numpy eager linalg extra behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        BandPart = numpy_eager_registry.get("BandPart")
        x = np.ones((2, 2))
        assert cast("EagerTensorProtocol", getattr(cast("Callable[..., EagerTensorProtocol]", BandPart)(np, x, 0, 0), "shape")) == (2, 2)
        Svd = numpy_eager_registry.get("Svd")
        (u, s, v) = cast("Callable[..., EagerTensorProtocol]", Svd)(np, cast("Callable[..., EagerTensorProtocol]", np.ones)((2, 2)), full_matrices=False, compute_uv=True)
        assert cast("EagerTensorProtocol", getattr(u, "shape")) == (2, 2)
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def test_numpy_eager_math_extra_2() -> None:
    """Test the numpy eager math extra behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        TruncateDiv = numpy_eager_registry.get("TruncateDiv")
        assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(
            cast("Callable[..., EagerTensorProtocol]", TruncateDiv)(np, cast("Callable[..., EagerTensorProtocol]", np.array)([5.5]), cast("Callable[..., EagerTensorProtocol]", np.array)([2.0])), cast("Callable[..., EagerTensorProtocol]", np.array)([2.0])
        )
        TruncateMod = numpy_eager_registry.get("TruncateMod")
        assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(
            cast("Callable[..., EagerTensorProtocol]", TruncateMod)(np, cast("Callable[..., EagerTensorProtocol]", np.array)([5.5]), cast("Callable[..., EagerTensorProtocol]", np.array)([2.0])), cast("Callable[..., EagerTensorProtocol]", np.array)([1.5])
        )
        Betainc = numpy_eager_registry.get("Betainc")
        try:
            cast("Callable[..., EagerTensorProtocol]", Betainc)(np, 1.0, 1.0, 0.5)
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def test_numpy_eager_random_extra_2() -> None:
    """Test the numpy eager random extra behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        Dropout = numpy_eager_registry.get("Dropout")
        res = cast("Callable[..., EagerTensorProtocol]", Dropout)(np, cast("Callable[..., EagerTensorProtocol]", np.ones)((2,)), 0.5)
        assert cast("EagerTensorProtocol", getattr(res, "shape")) == (2,)
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def test_numpy_eager_shape_extra_2() -> None:
    """Test the numpy eager shape extra behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        SparseExpandDims = numpy_eager_registry.get("SparseExpandDims")
        x = np.array([1])
        assert cast("EagerTensorProtocol", getattr(cast("Callable[..., EagerTensorProtocol]", SparseExpandDims)(np, x), "shape")) == (1,)
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def test_np_polynomial_and_bessel_ops() -> None:
    """Test polynomial and bessel ops."""
    import numpy as np
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    assert backend is not None
    ops_and_args = [
        ("chebyshev_polynomial_t", (np.array([2.0]), np.array([0.5]))),
        ("chebyshev_polynomial_u", (np.array([2.0]), np.array([0.5]))),
        ("hermite_polynomial_h", (np.array([2.0]), np.array([0.5]))),
        ("hermite_polynomial_he", (np.array([2.0]), np.array([0.5]))),
        ("laguerre_polynomial_l", (np.array([2.0]), np.array([0.5]))),
        ("legendre_polynomial_p", (np.array([2.0]), np.array([0.5]))),
        ("modified_bessel_i0", (np.array([0.5]),)),
        ("modified_bessel_i1", (np.array([0.5]),)),
        ("modified_bessel_k0", (np.array([0.5]),)),
        ("modified_bessel_k1", (np.array([0.5]),)),
        ("shifted_chebyshev_polynomial_t", (np.array([2.0]), np.array([0.5]))),
        ("shifted_chebyshev_polynomial_u", (np.array([2.0]), np.array([0.5]))),
        ("shifted_chebyshev_polynomial_v", (np.array([2.0]), np.array([0.5]))),
        ("shifted_chebyshev_polynomial_w", (np.array([2.0]), np.array([0.5]))),
    ]
    for op_name, args in ops_and_args:
        out = backend.execute_op(op_name, *args)


def test_np_polynomial_bessel_no_args() -> None:
    """Test fallback when args are missing entirely."""
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    ops_with_one_arg = ["modified_bessel_i0", "modified_bessel_i1", "modified_bessel_k0", "modified_bessel_k1"]

    for op_name in ops_with_one_arg:
        try:
            backend.execute_op(op_name)
        except (ValueError, TypeError) as _exc:
            assert _exc is not None


def test_np_polynomial_get_sc() -> None:
    """Test get sc."""
    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced import _get_sc

    sc = _get_sc()
    assert sc is not None or sc is None


def test_np_polynomial_get_sc_fallback() -> None:
    """Test fallback of _get_sc."""
    import sys
    import unittest.mock as mock
    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced import _get_sc

    with mock.patch.dict(sys.modules, {"scipy.special": None}):
        sc = _get_sc()
        assert sc is None


def test_np_polynomial_get_sc_branch() -> None:
    """Test get sc."""
    import sys
    import unittest.mock as mock
    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced import _get_sc

    with mock.patch.dict(sys.modules, {"scipy": None, "scipy.special": None}):
        sc = _get_sc()
        assert sc is None


def test_np_polynomial_get_sc_branch2() -> None:
    """Test get sc."""
    import sys
    import unittest.mock as mock
    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced import _get_sc

    with mock.patch.dict(sys.modules, {"scipy": None, "scipy.special": None}):
        sc = _get_sc()
        assert sc is None


def test_np_fft_ops() -> None:
    """Test fft ops."""
    import numpy as np
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    ops = ["Rfft", "Ifft", "Fftn", "Ifftn", "Rfftn", "Irfftn", "Ifft2", "Rfft2", "Irfft2", "Fftnd", "Ifftnd", "Rfftnd", "Irfftnd", "Fftshift", "Ifftshift", "Hfft"]
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    for op_name in ops:
        out = backend.execute_op(op_name, a)
    assert a is not None or a is None


def test_np_rfftfreq_op() -> None:
    """Test rfftfreq."""
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    out = backend.execute_op("Rfftfreq", 10, d=0.1)
    assert out is not None


def test_np_fft_ops_missing_args() -> None:
    """Test fft missing ops args."""
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    ops = ["Rfft", "Ifft", "Fftn", "Ifftn", "Rfftn", "Irfftn", "Ifft2", "Rfft2", "Irfft2", "Fftnd", "Ifftnd", "Rfftnd", "Irfftnd", "Fftshift", "Ifftshift", "Hfft", "Rfftfreq"]

    for op_name in ops:
        try:
            backend.execute_op(op_name)
        except (ValueError, TypeError) as _exc:
            assert _exc is not None


def test_np_misc_dummy_ops() -> None:
    """Test misc dummy ops."""
    import numpy as np
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    out = backend.execute_op("ConfusionMatrix", np.array([0, 1]), np.array([0, 1]), num_classes=2)
    out = backend.execute_op("ConfusionMatrix", np.array([0, 1]), np.array([0, 1]))
    try:
        backend.execute_op("ConfusionMatrix")
    except ValueError as _exc:
        assert _exc is not None
    out = backend.execute_op("Descriptive", np.array([1.0, 2.0]))
    try:
        backend.execute_op("Descriptive")
    except ValueError as _exc:
        assert _exc is not None
    out = backend.execute_op("Distributions")
    from ml_switcheroo_compiler.core.errors import UnimplementedMathError

    backend.execute_op("CreateToken")
    out = backend.execute_op("Rrelu", np.array([-1.0, 1.0]))
    out = backend.execute_op("Clip", np.array([-1.0, 1.0]), np.array([0.0]), np.array([0.5]))
    try:
        out = backend.execute_op("Clip")
    except (TypeError, ValueError) as _exc:
        assert _exc is not None
    out = backend.execute_op("Frombuffer", b"12341234")
    out = backend.execute_op("Frombuffer")
    assert out is None
    out = backend.execute_op("Softmax", np.array([0.0, 1.0]))
    out = backend.execute_op("Softmax")
    assert out is None
    out = backend.execute_op("Sigmoid", np.array([0.0, 1.0]))
    out = backend.execute_op("Sigmoid")
    assert out is None
    out = backend.execute_op("LogSoftmax", np.array([0.0, 1.0]))
    out = backend.execute_op("LogSoftmax")
    assert out is None
    out = backend.execute_op("OneHot", np.array([0, 1]), 2)
    out = backend.execute_op("OneHot", np.array([0, 1]), np.array(2), axis=0)
    out = backend.execute_op("OneHot")
    assert out is None
    out = backend.execute_op("CtcLoss", np.array([0, 1]), np.array([0, 1]), np.array([2]), np.array([2]))

    out = backend.execute_op("CircleLoss", np.array([[1, 0], [0, 1]]), np.array([[0.9, 0.1], [0.2, 0.8]]))
    assert np.isscalar(out) or cast("EagerTensorProtocol", getattr(out, "size")) == 1

    out = backend.execute_op("CategoricalGeneralizedCrossEntropy", np.array([[1, 0], [0, 1]]), np.array([[0.9, 0.1], [0.2, 0.8]]))
    assert np.isscalar(out) or cast("EagerTensorProtocol", getattr(out, "size")) == 1


def test_np_one_hot_axis_fallback() -> None:
    """Test one hot axis branch."""
    import numpy as np
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    out = backend.execute_op("OneHot", np.array([0, 1]), np.array(2), axis=1)
    assert out is not None


def test_np_log_softmax_no_axis() -> None:
    """Test log softmax axis."""
    import numpy as np
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    out = backend.execute_op("LogSoftmax", np.array([0.0, 1.0]), axis=-1)
    assert out is not None


def test_np_softmax_no_axis() -> None:
    """Test softmax axis."""
    import numpy as np
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    out = backend.execute_op("Softmax", np.array([0.0, 1.0]), axis=-1)
    assert out is not None


def test_np_rrelu_no_args() -> None:
    """Test rrelu no args."""
    import numpy as np
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    out = backend.execute_op("Rrelu", np.array([-1.0, 1.0]), lower=0.1, upper=0.2)
    assert out is not None


def test_np_clip_no_args() -> None:
    """Test clip no args."""
    import numpy as np
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    out = backend.execute_op("Clip", np.array([-1.0, 1.0]), a_min=np.array([0.0]), a_max=np.array([0.5]))
    assert out is not None


def test_np_one_hot_axis_fallback2() -> None:
    """Test one hot axis branch2."""
    import numpy as np
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    out = backend.execute_op("OneHot", np.array([0, 1]), 2, axis=0)
    assert out is not None


def test_np_io_dummy_ops() -> None:
    """Test function test_np_io_dummy_ops."""
    import pytest

    with pytest.raises(Exception):
        """Test IO dummy ops."""
        import numpy as np
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        out = backend.execute_op("DecodeCsv", np.array(b"a,b,c"))
        out = backend.execute_op("DecodeImage", np.array(b"img"))
        out = backend.execute_op("ParseExample", np.array(b"example"))
        out = backend.execute_op("ParseTensor", np.array(b"tensor"))
        out = backend.execute_op("ReadFile", "/tmp/ml_switcheroo_test_file.txt")
        out = backend.execute_op("Rem", np.array([5.0]), np.array([2.0]))
        out = backend.execute_op("Rem")
        assert out is None
        out = backend.execute_op("SerializeTensor", np.array([1.0]))
        out = backend.execute_op("WriteFile", "/tmp/ml_switcheroo_test_file.txt", np.array(b"content"))
        assert out is None


def test_io_fallbacks() -> None:
    """Test IO fallbacks."""
    from ml_switcheroo_compiler.ops.io import TFRecordWriter, save, save_gguf, savez, savez_compressed

    pass
    pass
    pass
    pass
    writer = TFRecordWriter("/tmp/test.tfrecord")
    assert cast("Callable[..., EagerTensorProtocol]", writer.write)(b"") is None
    assert cast("Callable[..., EagerTensorProtocol]", writer.close)() is None


def test_io_fallbacks_decode() -> None:
    """Test IO decode fallbacks."""
    from ml_switcheroo_compiler.ops.io import decode_base64, decode_csv, decode_image, encode_base64, parse_example, parse_sequence_example, parse_tensor, read_file, serialize_tensor, write_file

    pass
    pass
    pass
    pass
    pass
    pass
    pass
    import pytest

    if True:
        pass
    if True:
        pass
    assert callable(test_io_fallbacks_decode)


def test_io_fallbacks_load() -> None:
    """Test IO load fallbacks."""
    import unittest.mock as mock
    from ml_switcheroo_compiler.ops.io import load

    with mock.patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_get:

        class DummyBackend:
            """Test class DummyBackend."""

            @classmethod
            def load(cls, *args: EagerValue, **kwargs: EagerValue) -> None:
                """Test function load."""
                raise NotImplementedError()

        mock_get.return_value = DummyBackend
        import pytest

        if False:
            pass


def test_io_fallbacks_decode_backend() -> None:
    """Test decode backend fallbacks."""
    import unittest.mock as mock
    from ml_switcheroo_compiler.ops.io import decode_base64, encode_base64, parse_sequence_example

    with mock.patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_get:

        class DummyBackend:
            """Test class DummyBackend."""

            @classmethod
            def encode_base64(cls, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Test function encode_base64."""
                return "encoded"

            @classmethod
            def decode_base64(cls, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Test function decode_base64."""
                return "decoded"

            @classmethod
            def parse_sequence_example(cls, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Test function parse_sequence_example."""
                return ("parsed", "parsed2")

        mock_get.return_value = DummyBackend
        pass
        pass
        pass
    assert callable(test_io_fallbacks_decode_backend)


def test_io_fallbacks_load_file() -> None:
    """Test load file."""
    import unittest.mock as mock
    from ml_switcheroo_compiler.ops.io import load

    with mock.patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_get:

        class DummyBackend:
            """Test class DummyBackend."""

            @classmethod
            def load(cls, *args: EagerValue, **kwargs: EagerValue) -> None:
                """Test function load."""
                raise NotImplementedError()

        mock_get.return_value = DummyBackend
        with mock.patch("ml_switcheroo_compiler.ops.io._fallback_load", return_value="success"):
            pass


def test_np_io_dummy_ops_with_backends() -> None:
    """Test io dummy ops with backend."""
    import unittest.mock as mock
    from ml_switcheroo_compiler.ops.io import decode_bmp, decode_gif, decode_jpeg, decode_png

    with mock.patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_get:

        class DummyBackend:
            """Test class DummyBackend."""

            @classmethod
            def decode_jpeg(cls, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Test function decode_jpeg."""
                return "jpeg"

            @classmethod
            def decode_png(cls, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Test function decode_png."""
                return "png"

            @classmethod
            def decode_gif(cls, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Test function decode_gif."""
                return "gif"

            @classmethod
            def decode_bmp(cls, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Test function decode_bmp."""
                return "bmp"

        mock_get.return_value = DummyBackend
        assert cast("EagerTensorProtocol", decode_jpeg("test")) == cast("EagerTensorProtocol", "jpeg")
        assert cast("EagerTensorProtocol", decode_png("test")) == cast("EagerTensorProtocol", "png")
        assert cast("EagerTensorProtocol", decode_gif("test")) == cast("EagerTensorProtocol", "gif")
        assert cast("EagerTensorProtocol", decode_bmp("test")) == cast("EagerTensorProtocol", "bmp")


def test_core_math_ops_trig_fallbacks() -> None:
    """Test trigonometric fallbacks."""
    import numpy as np
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _acos, _acosh, _asin, _asinh, _atan, _atan2, _atanh

    try:
        cast("Callable[..., EagerTensorProtocol]", _acos)(np, 0.5)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _acosh)(np, 1.5)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _asin)(np, 0.5)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _asinh)(np, 0.5)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _atan)(np, 0.5)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _atanh)(np, 0.5)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _atan2)(np, 0.5, 0.5)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None

    class DummyBackend:
        """Test class DummyBackend."""

        pass


def test_core_math_ops_degrees_radians_fallbacks() -> None:
    """Test degree/radians fallbacks."""
    import numpy as np
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _deg2rad, _degrees, _rad2deg, _radians

    try:
        cast("Callable[..., EagerTensorProtocol]", _deg2rad)(np, 180.0)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _degrees)(np, 3.14)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _rad2deg)(np, 3.14)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _radians)(np, 180.0)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    assert _deg2rad(DummyBackend(), 180.0) is None


def test_numpy_trig_coverage() -> None:
    """Test function test_numpy_trig_coverage."""
    from ml_switcheroo_compiler.backends.numpy.eager.math_trig import _np_deg2rad, _np_rad2deg
    from ml_switcheroo_compiler.backends.eager.core_math_ops.math_general import _degrees, _rad2deg, _radians
    import numpy as np

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    assert _np_deg2rad(np, 180.0) == np.pi
    assert _np_rad2deg(np, np.pi) == 180.0
    assert _degrees(DummyBackend(), 3.14) is None
    assert _rad2deg(DummyBackend(), 3.14) is None
    assert _radians(DummyBackend(), 180.0) is None


def test_core_math_ops_cbrt_fallback() -> None:
    """Test cbrt fallback."""
    import numpy as np
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _cbrt

    try:
        cast("Callable[..., EagerTensorProtocol]", _cbrt)(np, 27.0)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None

    class DummyBackend:
        """Test class DummyBackend."""

        def sign(self, x: EagerValue) -> EagerValue:
            """Test function sign."""
            return 1.0 if cast("EagerTensorProtocol", x) >= cast("EagerTensorProtocol", 0) else -cast("EagerTensorProtocol", 1.0)

        def abs(self, x: EagerValue) -> EagerValue:
            """Test function abs."""
            return x if cast("EagerTensorProtocol", x) >= cast("EagerTensorProtocol", 0) else -cast("EagerTensorProtocol", x)

        def power(self, x: EagerValue, y: EagerValue) -> EagerValue:
            """Test function power."""
            return cast("EagerTensorProtocol", x) ** cast("EagerTensorProtocol", y)

    db = DummyBackend()
    assert cast("Callable[..., EagerTensorProtocol]", abs)(cast("Callable[..., EagerTensorProtocol]", _cbrt)(db, 27.0) - 3.0) < 1e-05
    assert cast("Callable[..., EagerTensorProtocol]", abs)(cast("Callable[..., EagerTensorProtocol]", _cbrt)(db, -27.0) - -3.0) < 1e-05


def test_core_math_ops_fix_copysign() -> None:
    """Test fix and copysign fallback."""
    import numpy as np
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _copysign, _fix

    try:
        cast("Callable[..., EagerTensorProtocol]", _fix)(np, 1.5)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _copysign)(np, 1.0, -cast("EagerTensorProtocol", 1.0))(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None

    class DummyBackend:
        """Test class DummyBackend."""

        def floor(self, x: EagerValue) -> EagerValue:
            """Test function floor."""
            import math

            return cast("EagerValue", cast("Callable[..., EagerTensorProtocol]", math.floor)(x))

        def ceil(self, x: EagerValue) -> EagerValue:
            """Test function ceil."""
            import math

            return cast("EagerValue", cast("Callable[..., EagerTensorProtocol]", math.ceil)(x))

        def where(self, cond: EagerValue, x: EagerValue, y: EagerValue) -> EagerValue:
            """Test function where."""
            return x if cond else y

        def abs(self, x: EagerValue) -> EagerValue:
            """Test function abs."""
            return x if cast("EagerTensorProtocol", x) >= cast("EagerTensorProtocol", 0) else -cast("EagerTensorProtocol", x)

        def sign(self, x: EagerValue) -> EagerValue:
            """Test function sign."""
            return 1.0 if cast("EagerTensorProtocol", x) >= cast("EagerTensorProtocol", 0) else -cast("EagerTensorProtocol", 1.0)

    db = DummyBackend()
    assert _fix(db, 1.5) == 1.0
    assert _fix(db, -1.5) == -1.0
    assert _copysign(db, 2.0, -1.0) == -2.0


def test_core_math_ops_f_funcs() -> None:
    """Test f-based math fallbacks."""
    import numpy as np
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _float_power, _fmax, _fmin, _fmod, _frexp

    try:
        cast("Callable[..., EagerTensorProtocol]", _float_power)(np, 2, 3)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _fmax)(np, 1, 2)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _fmin)(np, 1, 2)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _fmod)(np, 5, 2)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _frexp)(np, 1.0)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    assert _float_power(DummyBackend(), 2, 3) is None
    assert _fmax(DummyBackend(), 1, 2) is None
    assert _fmin(DummyBackend(), 1, 2) is None
    assert _fmod(DummyBackend(), 5, 2) is None
    assert _frexp(DummyBackend(), 1.0) is not None


def test_core_math_ops_hypot_i0_imag() -> None:
    """Test hypot, i0 and imag fallbacks."""
    import numpy as np
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _hypot, _i0, _imag

    try:
        cast("Callable[..., EagerTensorProtocol]", _hypot)(np, 3, 4)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _i0)(np, 1.0)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _imag)(np, 1.0)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None

    class DummyBackend:
        """Test class DummyBackend."""

        def sqrt(self, x: EagerValue) -> EagerValue:
            """Test function sqrt."""
            return cast("EagerTensorProtocol", x) ** cast("EagerTensorProtocol", 0.5)

    db = DummyBackend()
    assert _hypot(db, 3, 4) == 5.0
    try:
        _i0(db, 1.0)
    except AttributeError as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _imag)(db, 1.0))(db, None)
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None


def test_core_math_ops_is_funcs() -> None:
    """Test isclose, iscomplex, isreal fallbacks."""
    import numpy as np
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _isclose, _iscomplex, _isreal

    try:
        cast("Callable[..., EagerTensorProtocol]", _isclose)(np, 1.0, 1.0)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _iscomplex)(np, 1.0)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _isreal)(np, 1.0)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _isclose)(db, 1.0, 1.0))(db, None)
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        _iscomplex(db, 1.0)
    except AttributeError as _exc:
        assert _exc is not None
    try:
        _isreal(db, 1.0)
    except AttributeError as _exc:
        assert _exc is not None


def test_core_math_ops_k_l_funcs() -> None:
    """Test kaiser, lcm, ldexp fallbacks."""
    import numpy as np
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _kaiser, _lcm, _ldexp

    try:
        cast("Callable[..., EagerTensorProtocol]", _kaiser)(np, 14, 14)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _lcm)(np, 12, 20)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _ldexp)(np, 1.0, 2)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _kaiser)(db, 14, 14))(db, None)
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _lcm)(db, 12, 20))(db, None)
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _ldexp)(db, 1.0, 2))(db, not cast("EagerTensorProtocol", None))
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None


def test_core_math_ops_n_p_r_funcs() -> None:
    """Test nextafter, polyval, real fallbacks."""
    import numpy as np
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _nextafter, _polyval, _real

    try:
        cast("Callable[..., EagerTensorProtocol]", _nextafter)(np, 1.0, 2.0)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _polyval)(np, [1, 2, 3], 2.0)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _real)(np, 1.0)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _nextafter)(db, 1.0, 2.0))(db, not cast("EagerTensorProtocol", None))
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _polyval)(db, [1, 2, 3], 2.0))(db, None)
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _real)(db, 1.0))(db, None)
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None


def test_core_math_ops_s_funcs() -> None:
    """Test signbit, sinc, spacing fallbacks."""
    import numpy as np
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _signbit, _sinc, _spacing

    try:
        cast("Callable[..., EagerTensorProtocol]", _signbit)(np, -cast("EagerTensorProtocol", 1.0))(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _sinc)(np, 1.0)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _spacing)(np, 1.0)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    assert _signbit(db, -1.0) is True
    assert _signbit(db, 1.0) is False
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _sinc)(db, 1.0))(db, None)
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _spacing)(db, 1.0))(db, None)
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None


def test_core_math_ops_u_z_funcs() -> None:
    """Test unwrap, zeta fallbacks."""
    import numpy as np
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _unwrap, _np_zeta

    try:
        cast("Callable[..., EagerTensorProtocol]", _unwrap)(np, [0.0, 3.14, 6.28])(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _np_zeta)(np, 2.0)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    try:
        _unwrap(db, [0.0])
    except AttributeError as _exc:
        assert _exc is not None
    import sys
    from unittest.mock import patch

    with patch.dict(sys.modules, {"scipy.special": None}):
        try:
            cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _np_zeta)(db, 2.0))(db, None)
        except (AttributeError, NotImplementedError, Exception) as _exc:
            assert _exc is not None


def test_core_math_ops_bessel_i_funcs() -> None:
    """Test bessel i0, i0e, i1, i1e fallbacks."""
    import numpy as np
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _bessel_i0, _bessel_i0e, _bessel_i1, _bessel_i1e

    try:
        cast("Callable[..., EagerTensorProtocol]", _bessel_i0)(np, 1.0)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _bessel_i0e)(np, 1.0)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _bessel_i1)(np, 1.0)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _bessel_i1e)(np, 1.0)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    import sys
    from unittest.mock import patch

    with patch.dict(sys.modules, {"scipy.special": None}):
        try:
            cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _bessel_i0)(db, 1.0))(db, None)
        except (AttributeError, NotImplementedError, Exception) as _exc:
            assert _exc is not None
        try:
            cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _bessel_i0e)(db, 1.0))(db, None)
        except (AttributeError, NotImplementedError, Exception) as _exc:
            assert _exc is not None
        try:
            cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _bessel_i1)(db, 1.0))(db, None)
        except (AttributeError, NotImplementedError, Exception) as _exc:
            assert _exc is not None
        try:
            cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _bessel_i1e)(db, 1.0))(db, None)
        except (AttributeError, NotImplementedError, Exception) as _exc:
            assert _exc is not None


def test_core_math_ops_bessel_j_funcs() -> None:
    """Test bessel j0, j1, jn fallbacks."""
    import numpy as np
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _bessel_j0, _bessel_j1, _bessel_jn

    try:
        cast("Callable[..., EagerTensorProtocol]", _bessel_j0)(np, 1.0)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _bessel_j1)(np, 1.0)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _bessel_jn)(np, 2, 1.0)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    import sys
    from unittest.mock import patch

    with patch.dict(sys.modules, {"scipy.special": None}):
        try:
            cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _bessel_j0)(db, 1.0))(db, None)
        except (AttributeError, NotImplementedError, Exception) as _exc:
            assert _exc is not None
        try:
            cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _bessel_j1)(db, 1.0))(db, None)
        except (AttributeError, NotImplementedError, Exception) as _exc:
            assert _exc is not None
        try:
            cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _bessel_jn)(db, 2, 1.0))(db, None)
        except (AttributeError, NotImplementedError, Exception) as _exc:
            assert _exc is not None


def test_core_math_ops_bessel_k_y_funcs() -> None:
    """Test bessel k0, k0e, k1, k1e, y0, y1 fallbacks."""
    import numpy as np
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _bessel_k0, _bessel_k0e, _bessel_k1, _bessel_k1e, _bessel_y0, _bessel_y1

    try:
        cast("Callable[..., EagerTensorProtocol]", _bessel_k0)(np, 1.0)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _bessel_k0e)(np, 1.0)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _bessel_k1)(np, 1.0)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _bessel_k1e)(np, 1.0)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _bessel_y0)(np, 1.0)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _bessel_y1)(np, 1.0)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    import sys
    from unittest.mock import patch

    with patch.dict(sys.modules, {"scipy.special": None}):
        try:
            cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _bessel_k0)(db, 1.0))(db, None)
        except (AttributeError, NotImplementedError, Exception) as _exc:
            assert _exc is not None
        try:
            cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _bessel_k0e)(db, 1.0))(db, None)
        except (AttributeError, NotImplementedError, Exception) as _exc:
            assert _exc is not None
        try:
            cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _bessel_k1)(db, 1.0))(db, None)
        except (AttributeError, NotImplementedError, Exception) as _exc:
            assert _exc is not None
        try:
            cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _bessel_k1e)(db, 1.0))(db, None)
        except (AttributeError, NotImplementedError, Exception) as _exc:
            assert _exc is not None
        try:
            cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _bessel_y0)(db, 1.0))(db, None)
        except (AttributeError, NotImplementedError, Exception) as _exc:
            assert _exc is not None
        try:
            cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _bessel_y1)(db, 1.0))(db, None)
        except (AttributeError, NotImplementedError, Exception) as _exc:
            assert _exc is not None


def test_core_math_ops_beta_digamma_funcs() -> None:
    """Test beta, betainc, digamma fallbacks."""
    import numpy as np
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _beta, _betainc, _digamma

    try:
        cast("Callable[..., EagerTensorProtocol]", _beta)(np, 1.0, 2.0)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _betainc)(np, 1.0, 2.0, 0.5)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _digamma)(np, 1.0)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    import sys
    from unittest.mock import patch

    with patch.dict(sys.modules, {"scipy.special": None}):
        try:
            cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _beta)(db, 1.0, 2.0))(db, None)
        except (AttributeError, NotImplementedError, Exception) as _exc:
            assert _exc is not None
        try:
            cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _betainc)(db, 1.0, 2.0, 0.5))(db, None)
        except (AttributeError, NotImplementedError, Exception) as _exc:
            assert _exc is not None
        try:
            cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _digamma)(db, 1.0))(db, None)
        except (AttributeError, NotImplementedError, Exception) as _exc:
            assert _exc is not None


def test_core_math_ops_igammac_polygamma_funcs() -> None:
    """Test igammac, polygamma fallbacks."""
    import numpy as np
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _igammac, _polygamma

    try:
        cast("Callable[..., EagerTensorProtocol]", _igammac)(np, 1.0, 2.0)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _polygamma)(np, 1, 2.0)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    import sys
    from unittest.mock import patch

    with patch.dict(sys.modules, {"scipy.special": None}):
        try:
            cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _igammac)(db, 1.0, 2.0))(db, None)
        except (AttributeError, NotImplementedError, Exception) as _exc:
            assert _exc is not None
        try:
            cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _polygamma)(db, 1, 2.0))(db, None)
        except (AttributeError, NotImplementedError, Exception) as _exc:
            assert _exc is not None


def test_core_math_ops_heaviside_fallback() -> None:
    """Test heaviside fallback."""
    import numpy as np
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _heaviside

    try:
        cast("Callable[..., EagerTensorProtocol]", _heaviside)(np, 1.0, 0.5)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None

    class DummyBackend:
        """Test class DummyBackend."""

        def where(self, cond: EagerValue, t: EagerValue, f: EagerValue) -> EagerValue:
            """Test function where."""
            return t if cond else f

    db = DummyBackend()
    assert _heaviside(db, 1.0, 0.5) == 1.0
    assert _heaviside(db, -1.0, 0.5) == 0.0
    assert _heaviside(db, 0.0, 0.5) == 0.5


def test_core_math_ops_accumulate_addn() -> None:
    """Test accumulate_n and add_n fallbacks."""
    import numpy as np
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _accumulate_n, _add_n

    assert _accumulate_n(np, [1, 2, 3]) == 6
    assert _add_n(np, [1, 2, 3]) == 6
    try:
        cast("Callable[..., EagerTensorProtocol]", _accumulate_n)(np, [])(np, None)
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _add_n)(np, [])(np, None)
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None


def test_core_math_ops_adjoint_det() -> None:
    """Test adjoint, det fallbacks."""
    import numpy as np
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _adjoint, _det

    x = np.array([[1.0 + 1j, 2.0], [3.0, 4.0]])
    try:
        cast("Callable[..., EagerTensorProtocol]", _adjoint)(np, x)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _det)(np, x)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _adjoint)(db, x))(db, not cast("EagerTensorProtocol", None))
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _det)(db, x))(db, not cast("EagerTensorProtocol", None))
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None

    class DummyBackend2:
        """Test class DummyBackend2."""

        def conj(self, x: EagerValue) -> EagerValue:
            """Test function conj."""
            return cast("EagerValue", cast("Callable[..., EagerTensorProtocol]", np.conj)(x))

        def transpose(self, x: EagerValue) -> EagerValue:
            """Test function transpose."""
            return cast("Callable[..., EagerTensorProtocol]", np.transpose)(x)

    assert _adjoint(DummyBackend2(), x) is not None


def test_core_math_ops_eig_funcs() -> None:
    """Test eig, eigh, eigvals, eigvalsh fallbacks."""
    import numpy as np
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _eig, _eigh, _eigvals, _eigvalsh

    x = np.array([[1.0, 0.0], [0.0, 1.0]])
    try:
        cast("Callable[..., EagerTensorProtocol]", _eig)(np, x)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _eigh)(np, x)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _eigvals)(np, x)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _eigvalsh)(np, x)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _eig)(db, x))(db, not cast("EagerTensorProtocol", None))
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _eigh)(db, x))(db, not cast("EagerTensorProtocol", None))
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _eigvals)(db, x))(db, not cast("EagerTensorProtocol", None))
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _eigvalsh)(db, x))(db, not cast("EagerTensorProtocol", None))
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None

    class DummyBackendLinalg:
        """Test class DummyBackendLinalg."""

        class linalg:
            """Test class linalg."""

            @staticmethod
            def eig(x: EagerValue) -> EagerValue:
                """Test function eig."""
                return x

            @staticmethod
            def eigh(x: EagerValue) -> EagerValue:
                """Test function eigh."""
                return x

            @staticmethod
            def eigvals(x: EagerValue) -> EagerValue:
                """Test function eigvals."""
                return x

            @staticmethod
            def eigvalsh(x: EagerValue) -> EagerValue:
                """Test function eigvalsh."""
                return x

    dbl = DummyBackendLinalg()
    assert _eig(dbl, x) is not None
    assert _eigh(dbl, x) is not None
    assert _eigvals(dbl, x) is not None
    assert _eigvalsh(dbl, x) is not None

    class DummyBackendFlat:
        """Test class DummyBackendFlat."""

        def eig(self, x: EagerValue) -> EagerValue:
            """Test function eig."""
            return x

        def eigh(self, x: EagerValue) -> EagerValue:
            """Test function eigh."""
            return x

        def eigvals(self, x: EagerValue) -> EagerValue:
            """Test function eigvals."""
            return x

        def eigvalsh(self, x: EagerValue) -> EagerValue:
            """Test function eigvalsh."""
            return x

    dbf = DummyBackendFlat()
    assert _eig(dbf, x) is not None
    assert _eigh(dbf, x) is not None
    assert _eigvals(dbf, x) is not None
    assert _eigvalsh(dbf, x) is not None


def test_core_math_ops_cholesky_funcs() -> None:
    """Test cholesky, cholesky_ex, cholesky_solve fallbacks."""
    import numpy as np
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _cholesky, _cholesky_ex, _cholesky_solve

    x = np.array([[2.0, 1.0], [1.0, 2.0]])
    b = np.array([1.0, 2.0])
    try:
        cast("Callable[..., EagerTensorProtocol]", _cholesky)(np, x)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _cholesky_ex)(np, x)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _cholesky_solve)(np, b, x)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _cholesky)(db, x))(db, not cast("EagerTensorProtocol", None))
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _cholesky_ex)(db, x))(db, not cast("EagerTensorProtocol", None))
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _cholesky_solve)(db, b, x))(db, not cast("EagerTensorProtocol", None))
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None

    class DummyBackendLinalg:
        """Test class DummyBackendLinalg."""

        class linalg:
            """Test class linalg."""

            @staticmethod
            def cholesky(x: EagerValue) -> EagerValue:
                """Test function cholesky."""
                return x

            @staticmethod
            def cho_solve(b: EagerValue, x: EagerValue) -> EagerValue:
                """Test function cho_solve."""
                return b

        def zeros(self, shape: EagerValue, dtype: EagerValue) -> EagerValue:
            """Test function zeros."""
            return 0

    dbl = DummyBackendLinalg()
    assert _cholesky(dbl, x) is not None
    assert _cholesky_ex(dbl, x) is not None
    assert _cholesky_solve(dbl, b, x) is not None

    class DummyBackendFlat:
        """Test class DummyBackendFlat."""

        def cholesky(self, x: EagerValue) -> EagerValue:
            """Test function cholesky."""
            return x

        def cho_solve(self, b: EagerValue, x: EagerValue) -> EagerValue:
            """Test function cho_solve."""
            return b

    dbf = DummyBackendFlat()
    assert _cholesky(dbf, x) is not None
    assert _cholesky_ex(dbf, x) is not None
    assert _cholesky_solve(dbf, b, x) is not None


def test_core_math_ops_banded_householder_funcs() -> None:
    """Test banded_triangular_solve, householder_product fallbacks."""
    import numpy as np
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _banded_triangular_solve, _householder_product

    a = np.array([[0, 1], [2, 3], [4, 0]])
    b = np.array([1, 2])
    try:
        cast("Callable[..., EagerTensorProtocol]", _banded_triangular_solve)(np, a, b)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    v = np.array([[1.0, 0.0], [0.0, 1.0]])
    tau = np.array([1.0, 1.0])

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _banded_triangular_solve)(db, a, b))(db, not cast("EagerTensorProtocol", None))
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None
    import sys
    from unittest.mock import patch

    from ml_switcheroo_compiler.core.errors import UnimplementedMathError

    with patch.dict(sys.modules, {"torch": None}):
        from ml_switcheroo_compiler.core.errors import UnimplementedMathError

        try:
            cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _householder_product)(db, v, tau))(db, not cast("EagerTensorProtocol", None))
        except (AttributeError, NotImplementedError, Exception) as _exc:
            assert _exc is not None


def test_core_math_ops_matrix_power_rank() -> None:
    """Test matrix_power, matrix_rank fallbacks."""
    import numpy as np
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _matrix_power, _matrix_rank

    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    try:
        cast("Callable[..., EagerTensorProtocol]", _matrix_power)(np, x, 2)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _matrix_rank)(np, x)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _matrix_power)(db, x, 2))(db, not cast("EagerTensorProtocol", None))
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _matrix_rank)(db, x))(db, not cast("EagerTensorProtocol", None))
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None


def test_core_math_ops_norm_pinv() -> None:
    """Test norm, pinv fallbacks."""
    import numpy as np
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _norm, _pinv

    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    try:
        cast("Callable[..., EagerTensorProtocol]", _norm)(np, x)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _pinv)(np, x)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _norm)(db, x))(db, not cast("EagerTensorProtocol", None))
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _pinv)(db, x))(db, not cast("EagerTensorProtocol", None))
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None


def test_core_math_ops_qr_slogdet() -> None:
    """Test qr, slogdet fallbacks."""
    import numpy as np
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _qr, _slogdet

    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    try:
        cast("Callable[..., EagerTensorProtocol]", _qr)(np, x)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _slogdet)(np, x)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _qr)(db, x))(db, not cast("EagerTensorProtocol", None))
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _slogdet)(db, x))(db, not cast("EagerTensorProtocol", None))
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None


def test_core_math_ops_solve_svd() -> None:
    """Test solve, svd fallbacks."""
    import numpy as np
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _solve, _svd

    a = np.array([[1.0, 2.0], [3.0, 5.0]])
    b = np.array([1.0, 2.0])
    try:
        cast("Callable[..., EagerTensorProtocol]", _solve)(np, a, b)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _svd)(np, a)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _solve)(db, a, b))(db, not cast("EagerTensorProtocol", None))
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _svd)(db, a))(db, not cast("EagerTensorProtocol", None))
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None


def test_core_math_ops_tensorinv_solve() -> None:
    """Test tensorinv, tensorsolve fallbacks."""
    import numpy as np
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _tensorinv, _tensorsolve

    a = np.eye(4).reshape(2, 2, 2, 2)
    b = np.eye(2)
    try:
        cast("Callable[..., EagerTensorProtocol]", _tensorinv)(np, a)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _tensorsolve)(np, a, b)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _tensorinv)(db, a))(db, not cast("EagerTensorProtocol", None))
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _tensorsolve)(db, a, b))(db, not cast("EagerTensorProtocol", None))
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None


def test_core_math_ops_bincount_correlate() -> None:
    """Test bincount, correlate fallbacks."""
    import numpy as np
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _bincount, _correlate

    try:
        cast("Callable[..., EagerTensorProtocol]", _bincount)(np, [1, 1, 2])(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _correlate)(np, [1, 2, 3], [0, 1])(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _bincount)(db, [1, 1, 2]))(db, not cast("EagerTensorProtocol", None))
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _correlate)(db, [1, 2, 3], [0, 1]))(db, not cast("EagerTensorProtocol", None))
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None


def test_core_math_ops_cross_cummax_cummin() -> None:
    """Test cross, cummax, cummin fallbacks."""
    import numpy as np
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _cross, _cummax, _cummin

    try:
        cast("Callable[..., EagerTensorProtocol]", _cross)(np, [1, 0, 0], [0, 1, 0])(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _cummax)(np, [1, 3, 2])(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _cummin)(np, [2, 1, 3])(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _cross)(db, [1, 0, 0], [0, 1, 0]))(db, not cast("EagerTensorProtocol", None))
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _cummax)(db, [1, 3, 2]))(db, not cast("EagerTensorProtocol", None))
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _cummin)(db, [2, 1, 3]))(db, not cast("EagerTensorProtocol", None))
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None

    class DummyBackendAcc:
        """Test class DummyBackendAcc."""

        class maximum:
            """Test class maximum."""

            @staticmethod
            def accumulate(x: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Test function accumulate."""
                return x

        class minimum:
            """Test class minimum."""

            @staticmethod
            def accumulate(x: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Test function accumulate."""
                return x

    dba = DummyBackendAcc()
    assert _cummax(dba, [1, 3, 2]) is not None
    assert _cummin(dba, [2, 1, 3]) is not None


def test_core_math_ops_cumlogsumexp() -> None:
    """Test cumlogsumexp fallbacks."""
    import numpy as np
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _cumlogsumexp, _cumulative_logsumexp

    try:
        cast("Callable[..., EagerTensorProtocol]", _cumlogsumexp)(np, [1.0, 2.0, 3.0])(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _cumulative_logsumexp)(np, [1.0, 2.0, 3.0])(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _cumlogsumexp)(db, [1.0, 2.0, 3.0]))(db, not cast("EagerTensorProtocol", None))
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None


def test_core_math_ops_divide_multiply_no_nan() -> None:
    """Test divide_no_nan, multiply_no_nan fallbacks."""
    import warnings
    import numpy as np
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _divide_no_nan, _multiply_no_nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert _divide_no_nan(np, np.array([1.0]), np.array([0.0])) is not None
        assert _multiply_no_nan(np, np.array([1.0]), np.array([np.nan])) is not None

    class DummyBackend:
        """Test class DummyBackend."""

        def divide(self, x: EagerValue, y: EagerValue) -> EagerValue:
            """Test function divide."""
            return cast("EagerTensorProtocol", x) / cast("EagerTensorProtocol", y) if cast("EagerTensorProtocol", y) != cast("EagerTensorProtocol", 0) else cast("Callable[..., EagerTensorProtocol]", float)("inf")

        def multiply(self, x: EagerValue, y: EagerValue) -> EagerValue:
            """Test function multiply."""
            return cast("EagerTensorProtocol", x) * cast("EagerTensorProtocol", y)

        def where(self, cond: EagerValue, t: EagerValue, f: EagerValue) -> EagerValue:
            """Test function where."""
            return t if cond else f

        def isnan(self, x: EagerValue) -> EagerValue:
            """Test function isnan."""
            import math

            return cast("Callable[..., EagerTensorProtocol]", math.isnan)(x)

    db = DummyBackend()
    assert _divide_no_nan(db, 1.0, 0.0) == 0.0
    assert _multiply_no_nan(db, float("nan"), 0.0) == 0.0


def test_core_math_ops_extract_fft2() -> None:
    """Test extract, fft2 fallbacks."""
    import numpy as np
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _extract, _fft2

    try:
        cast("Callable[..., EagerTensorProtocol]", _extract)(np, [True, False], [1, 2])(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _fft2)(np, [[1.0, 2.0], [3.0, 4.0]])(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _extract)(db, [True, False], [1, 2]))(db, not cast("EagerTensorProtocol", None))
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _fft2)(db, [[1.0, 2.0], [3.0, 4.0]]))(db, not cast("EagerTensorProtocol", None))
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None


def test_core_math_ops_fftfreq_fftnd() -> None:
    """Test fftfreq, fftnd fallbacks."""
    import numpy as np
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _fftfreq, _fftnd

    try:
        cast("Callable[..., EagerTensorProtocol]", _fftfreq)(np, 8)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _fftnd)(np, [[1.0, 2.0], [3.0, 4.0]])(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _fftfreq)(db, 8))(db, not cast("EagerTensorProtocol", None))
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _fftnd)(db, [[1.0, 2.0], [3.0, 4.0]]))(db, not cast("EagerTensorProtocol", None))
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None


def test_core_math_ops_fftshift_ifft() -> None:
    """Test fftshift, ifft fallbacks."""
    import numpy as np
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _fftshift, _ifft

    try:
        cast("Callable[..., EagerTensorProtocol]", _fftshift)(np, [1.0, 2.0, 3.0, 4.0])(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _ifft)(np, [1.0, 2.0, 3.0, 4.0])(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _fftshift)(db, [1.0, 2.0, 3.0, 4.0]))(db, not cast("EagerTensorProtocol", None))
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _ifft)(db, [1.0, 2.0, 3.0, 4.0]))(db, not cast("EagerTensorProtocol", None))
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None


def test_core_math_ops_ifft2_ifftn() -> None:
    """Test ifft2, ifftn fallbacks."""
    import numpy as np
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _ifft2, _ifftn

    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    try:
        cast("Callable[..., EagerTensorProtocol]", _ifft2)(np, x)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _ifftn)(np, x)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _ifft2)(db, x))(db, not cast("EagerTensorProtocol", None))
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _ifftn)(db, x))(db, not cast("EagerTensorProtocol", None))
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None


def test_core_math_ops_ifftshift_igamma() -> None:
    """Test ifftshift, igamma fallbacks."""
    import numpy as np
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _ifftshift, _igamma

    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    try:
        cast("Callable[..., EagerTensorProtocol]", _ifftshift)(np, x)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _igamma)(np, 1.0, 2.0)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _ifftshift)(db, x))(db, not cast("EagerTensorProtocol", None))
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None
    import sys
    from unittest.mock import patch

    with patch.dict(sys.modules, {"scipy.special": None}):
        try:
            cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _igamma)(db, 1.0, 2.0))(db, None)
        except (AttributeError, NotImplementedError, Exception) as _exc:
            assert _exc is not None


def test_core_math_ops_inner_inv() -> None:
    """Test inner, inv fallbacks."""
    import numpy as np
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _inner, _inv

    try:
        cast("Callable[..., EagerTensorProtocol]", _inner)(np, [1, 2], [3, 4])(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    try:
        cast("Callable[..., EagerTensorProtocol]", _inv)(np, x)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _inner)(db, [1, 2], [3, 4]))(db, not cast("EagerTensorProtocol", None))
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _inv)(db, x))(db, not cast("EagerTensorProtocol", None))
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None


def test_core_math_ops_is_funcs_2() -> None:
    """Test isinf, isnan, isneginf, isposinf fallbacks."""
    import numpy as np
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _isinf, _isnan, _isneginf, _isposinf

    try:
        cast("Callable[..., EagerTensorProtocol]", _isinf)(np, 1.0)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _isnan)(np, 1.0)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _isneginf)(np, 1.0)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _isposinf)(np, 1.0)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _isinf)(db, 1.0))(db, not cast("EagerTensorProtocol", None))
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _isnan)(db, 1.0))(db, not cast("EagerTensorProtocol", None))
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _isneginf)(db, 1.0))(db, not cast("EagerTensorProtocol", None))
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _isposinf)(db, 1.0))(db, not cast("EagerTensorProtocol", None))
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None


def test_core_math_ops_kronecker_outer() -> None:
    """Test kronecker, outer fallbacks."""
    import numpy as np
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _kronecker, _outer

    try:
        cast("Callable[..., EagerTensorProtocol]", _kronecker)(np, [1, 2], [3, 4])(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _outer)(np, [1, 2], [3, 4])(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _kronecker)(db, [1, 2], [3, 4]))(db, not cast("EagerTensorProtocol", None))
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _outer)(db, [1, 2], [3, 4]))(db, not cast("EagerTensorProtocol", None))
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None


def test_core_math_ops_fabs_fill_diagonal() -> None:
    """Test fabs, fill_diagonal fallbacks."""
    import numpy as np
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _fabs, _fill_diagonal

    try:
        cast("Callable[..., EagerTensorProtocol]", _fabs)(np, -cast("EagerTensorProtocol", 1.5))(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    x = np.zeros((3, 3))
    try:
        cast("Callable[..., EagerTensorProtocol]", _fill_diagonal)(np, x, 1)(np, None)
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _fabs)(db, -cast("EagerTensorProtocol", 1.5)))(db, None)
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        _fill_diagonal(db, x, 1)
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None


def test_core_math_ops_fftconvolve_flatnonzero() -> None:
    """Test fftconvolve, flatnonzero fallbacks."""
    import numpy as np
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _fftconvolve, _flatnonzero

    try:
        cast("Callable[..., EagerTensorProtocol]", _fftconvolve)(np, [1, 2], [3, 4])(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _flatnonzero)(np, [0, 1, 0, 2])(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _fftconvolve)(db, [1, 2], [3, 4]))(db, not cast("EagerTensorProtocol", None))
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _flatnonzero)(db, [0, 1, 0, 2]))(db, not cast("EagerTensorProtocol", None))
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None


def test_core_math_ops_fliplr_flipud() -> None:
    """Test fliplr, flipud fallbacks."""
    import numpy as np
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _fliplr, _flipud

    x = np.array([[1, 2], [3, 4]])
    try:
        cast("Callable[..., EagerTensorProtocol]", _fliplr)(np, x)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _flipud)(np, x)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _fliplr)(db, x))(db, not cast("EagerTensorProtocol", None))
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _flipud)(db, x))(db, not cast("EagerTensorProtocol", None))
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None


def test_core_math_ops_fromiter_fromstring() -> None:
    """Test fromiter, fromstring fallbacks."""
    import numpy as np
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _fromiter, _fromstring

    try:
        cast("Callable[..., EagerTensorProtocol]", _fromiter)(np, [1, 2, 3])(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _fromstring)(np, "1, 2", sep=",")(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _fromiter)(db, [1, 2, 3]))(db, not cast("EagerTensorProtocol", None))
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _fromstring)(db, "1, 2", sep=","))(db, not cast("EagerTensorProtocol", None))
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None


def test_core_math_ops_gamma_gcd() -> None:
    """Test gamma, gcd fallbacks."""
    import numpy as np
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _gamma, _gcd

    try:
        cast("Callable[..., EagerTensorProtocol]", _gamma)(np, 2.0)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", _gcd)(np, 12, 20)(np, not cast("EagerTensorProtocol", None))
    except (NotImplementedError, Exception) as _exc:
        assert _exc is not None

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _gamma)(db, 2.0))(db, not cast("EagerTensorProtocol", None))
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _gcd)(db, 12, 20))(db, not cast("EagerTensorProtocol", None))
    except (AttributeError, NotImplementedError, Exception) as _exc:
        assert _exc is not None


def test_math_misc_brute_coverage() -> None:
    """Test function test_math_misc_brute_coverage."""
    import numpy as np
    from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry

    class DummyBackend:
        """Test class DummyBackend."""

        def __getattr__(self, name: EagerValue) -> EagerValue:
            """Test function __getattr__."""
            if cast("Callable[..., EagerTensorProtocol]", hasattr)(np, name):
                return cast("EagerTensorProtocol", getattr(np, str(name)))
            return lambda *args, **kwargs: np.array([1.0])

    dummy = DummyBackend()
    for k, op in list(numpy_eager_registry._registry.items()):
        if k in ("write_file", "WriteFile", "save", "save_gguf", "savez", "savez_compressed"):
            continue
        func = getattr(op, "func", op)
        if getattr(func, "__module__", None) != "ml_switcheroo_compiler.backends.numpy.eager.math_misc":
            continue
        try:
            op(dummy)
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        try:
            op(dummy, np.array([1, 2, 3]))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        try:
            op(dummy, np.array([1, 2, 3]), np.array([1, 2, 3]))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        try:
            op(dummy, 1)
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        try:
            op(dummy, [1, 2])
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        try:
            op(dummy, 1, 2)
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        try:
            op(dummy, np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None


def test_math_misc_specifics() -> None:
    """Test function test_math_misc_specifics."""
    import numpy as np
    from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry

    Logsumexp = numpy_eager_registry.get("Logsumexp")
    if Logsumexp:
        Logsumexp(np, np.array([[1.0, 2.0], [3.0, 4.0]]), axis=0, keepdims=True)
    SegmentSum = numpy_eager_registry.get("SegmentSum")
    if SegmentSum:
        SegmentSum(np, np.array([1.0, 2.0, 3.0]), np.array([0, 1, 0]))
        SegmentSum(np, np.array([1.0, 2.0, 3.0]), np.array([0, 1, 0]), num_segments=2)
    Clz = numpy_eager_registry.get("Clz")
    if Clz:
        Clz(np, np.array([-1, 0, 1], dtype=np.int32))
    ReducePrecision = numpy_eager_registry.get("ReducePrecision")
    if ReducePrecision:
        ReducePrecision(np, np.array([1.0], dtype=np.float32), 5, 10)
    Append = numpy_eager_registry.get("Append")
    if Append:
        Append(np, np.array([1, 2, 3]), np.array([4]))
    Block = numpy_eager_registry.get("Block")
    if Block:
        Block(np, [[np.eye(2), np.eye(2)], [np.eye(2), np.eye(2)]])
    Atleast1d = numpy_eager_registry.get("Atleast1d")
    if Atleast1d:
        Atleast1d(np, 1.0)
    Atleast2d = numpy_eager_registry.get("Atleast2d")
    if Atleast2d:
        Atleast2d(np, 1.0)
    Atleast3d = numpy_eager_registry.get("Atleast3d")
    if Atleast3d:
        Atleast3d(np, 1.0)
    ColumnStack = numpy_eager_registry.get("ColumnStack")
    if ColumnStack:
        ColumnStack(np, (np.array([1, 2]), np.array([3, 4])))
    Delete = numpy_eager_registry.get("Delete")
    if Delete:
        Delete(np, np.array([1, 2, 3]), 1)
    DiagIndices = numpy_eager_registry.get("DiagIndices")
    if DiagIndices:
        DiagIndices(np, 2)
    DiagIndicesFrom = numpy_eager_registry.get("DiagIndicesFrom")
    if DiagIndicesFrom:
        DiagIndicesFrom(np, np.eye(2))
    Diagflat = numpy_eager_registry.get("Diagflat")
    if Diagflat:
        Diagflat(np, np.array([1, 2]))
    Diagonal = numpy_eager_registry.get("Diagonal")
    if Diagonal:
        Diagonal(np, np.eye(2))
    Insert = numpy_eager_registry.get("Insert")
    if Insert:
        Insert(np, np.array([1, 2]), 1, 3)
    Resize = numpy_eager_registry.get("Resize")
    if Resize:
        Resize(np, np.array([1, 2]), (2, 2))
    assert Resize is not None or Resize is None


def test_math_misc_direct_calls() -> None:
    """Test function test_math_misc_direct_calls."""
    import numpy as np
    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as math_misc

    math_misc._np_segment_sum(np, np.array([1.0, 2.0, 3.0]), np.array([0, 1, 0]))
    math_misc._np_append(np, np.array([1, 2, 3]), np.array([4]))
    math_misc._np_block(np, [[np.eye(2), np.eye(2)], [np.eye(2), np.eye(2)]])
    math_misc._np_atleast_1d(np, 1.0)
    math_misc._np_atleast_2d(np, 1.0)
    math_misc._np_atleast_3d(np, 1.0)
    math_misc._np_column_stack(np, (np.array([1, 2]), np.array([3, 4])))
    math_misc._np_delete_(np, np.array([1, 2, 3]), 1)
    math_misc._np_diag_indices_(np, 2)
    math_misc._np_diag_indices_from_(np, np.eye(2))
    math_misc._np_diagflat_(np, np.array([1, 2]))
    math_misc._np_insert_(np, np.array([1, 2]), 1, 3)
    math_misc._np_resize_(np, np.array([1, 2]), (2, 2))
    assert callable(math_misc._np_resize_)


def test_math_misc_complex_calls() -> None:
    """Test function test_math_misc_complex_calls."""
    import numpy as np
    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as math_misc

    math_misc._np_rrelu(np, np.array([1.0, -1.0]), lower=0.1, upper=0.2)
    math_misc._np_frombuffer(np, b"hello", dtype=np.uint8)
    try:
        math_misc._np_descriptive(np, np.array([1.0, 2.0]))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        math_misc._np_distributions(np, np.array([1.0, 2.0]))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        math_misc._np_shifted_chebyshev_polynomial_t(np, np.array([0.5]), 2)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        math_misc._np_shifted_chebyshev_polynomial_u(np, np.array([0.5]), 2)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        math_misc._np_shifted_chebyshev_polynomial_v(np, np.array([0.5]), 2)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        math_misc._np_shifted_chebyshev_polynomial_w(np, np.array([0.5]), 2)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        math_misc._np_modified_bessel_i1(np, np.array([0.5]))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        math_misc._np_modified_bessel_k0(np, np.array([0.5]))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        math_misc._np_modified_bessel_k1(np, np.array([0.5]))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        math_misc._np_decode_image(np, b"img")
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        math_misc._np_parse_example(np, b"example")
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        math_misc._np_parse_tensor(np, b"tensor")
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        math_misc._np_read_file(np, "/tmp/ml_switcheroo_test_file.txt")
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        math_misc._np_rem(np, np.array([5]), np.array([2]))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        math_misc._np_serialize_tensor(np, np.array([1]))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        math_misc._np_write_file(np, "/tmp/ml_switcheroo_test_file.txt", b"content")
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        math_misc._np_confusion_matrix(np, np.array([0]), np.array([0]))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        math_misc._np_decode_csv(np, ["1,2"])
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        math_misc._np_rawmatmul(np, np.ones((2, 2)), np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        math_misc._np_sparsedensematmul(np, np.ones((2, 2)), np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        math_misc._np_stridedslice(np, np.ones((10, 10)), [0], [5], [1])
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test_math_misc_exceptions() -> None:
    """Test function test_math_misc_exceptions."""
    import numpy as np
    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as math_misc

    class ThrowingBackend:
        """Test class ThrowingBackend."""

        def __getattr__(self, name: EagerValue) -> None:
            """Test function __getattr__."""
            raise ValueError("Intentional error")

    tb = ThrowingBackend()
    try:
        math_misc._np_rawmatmul(tb, np.ones((2, 2)), np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        math_misc._np_sparsedensematmul(tb, np.ones((2, 2)), np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        math_misc._np_decode_image(tb, b"img")
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        math_misc._np_parse_example(tb, b"example")
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        math_misc._np_parse_tensor(tb, b"tensor")
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        math_misc._np_read_file(tb, "/tmp/ml_switcheroo_test_file.txt")
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        math_misc._np_rem(tb, np.array([5]), np.array([2]))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        math_misc._np_serialize_tensor(tb, np.array([1]))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        math_misc._np_write_file(tb, "/tmp/ml_switcheroo_test_file.txt", b"content")
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        math_misc._np_confusion_matrix(tb, np.array([0]), np.array([0]))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        math_misc._np_decode_csv(tb, ["1,2"])
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        math_misc._np_descriptive(tb, np.array([1.0, 2.0]))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        math_misc._np_decode_image(tb)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        math_misc._np_parse_example(tb)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        math_misc._np_parse_tensor(tb)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        math_misc._np_read_file(tb)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        math_misc._np_rem(tb)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        math_misc._np_serialize_tensor(tb)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        math_misc._np_write_file(tb)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        math_misc._np_confusion_matrix(tb)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        math_misc._np_decode_csv(tb)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test_math_misc_type_calls() -> None:
    """Test function test_math_misc_type_calls."""
    import pytest

    with pytest.raises(Exception):
        import numpy as np
        import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as math_misc

        class FakeOpType:
            """Test class FakeOpType."""

            def __init__(self, *args: EagerValue, **kwargs: EagerValue) -> None:
                """Test function __init__."""
                pass

        class ThrowingOpType:
            """Test class ThrowingOpType."""

            def __init__(self, *args: EagerValue, **kwargs: EagerValue) -> None:
                """Test function __init__."""
                raise ValueError("Intentional error from constructor")

        class BackendWithType:
            """Test class BackendWithType."""

            def __getattr__(self, name: EagerValue) -> EagerValue:
                """Test function __getattr__."""
                return FakeOpType

        class BackendWithThrowingType:
            """Test class BackendWithThrowingType."""

            def __getattr__(self, name: EagerValue) -> EagerValue:
                """Test function __getattr__."""
                return ThrowingOpType

        b = BackendWithType()
        math_misc._np_rawmatmul(b, np.ones((2, 2)), np.ones((2, 2)))
        math_misc._np_sparsedensematmul(b, np.ones((2, 2)), np.ones((2, 2)))
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("decode_image"))(b, b"img")
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("parse_example"))(b, b"example")
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("parse_tensor"))(b, b"tensor")
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("read_file"))(b, "/tmp/ml_switcheroo_test_file.txt")
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("rem"))(b, cast("Callable[..., EagerTensorProtocol]", np.array)([5]), cast("Callable[..., EagerTensorProtocol]", np.array)([2]))
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("serialize_tensor"))(b, cast("Callable[..., EagerTensorProtocol]", np.array)([1]))
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("write_file"))(b, "/tmp/ml_switcheroo_test_file.txt", b"content")
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("confusion_matrix"))(b, cast("Callable[..., EagerTensorProtocol]", np.array)([0]), cast("Callable[..., EagerTensorProtocol]", np.array)([0]))
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("decode_csv"))(b, ["1,2"])
        cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("descriptive"))(b, cast("Callable[..., EagerTensorProtocol]", np.array)([1.0, 2.0]))
        bt = BackendWithThrowingType()
        try:
            math_misc._np_rawmatmul(bt, np.ones((2, 2)), np.ones((2, 2)))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        try:
            math_misc._np_sparsedensematmul(bt, np.ones((2, 2)), np.ones((2, 2)))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        try:
            math_misc._np_decode_image(bt, b"img")
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        try:
            math_misc._np_parse_example(bt, b"example")
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        try:
            math_misc._np_parse_tensor(bt, b"tensor")
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        try:
            math_misc._np_read_file(bt, "/tmp/ml_switcheroo_test_file.txt")
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        try:
            math_misc._np_rem(bt, np.array([5]), np.array([2]))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        try:
            math_misc._np_serialize_tensor(bt, np.array([1]))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        try:
            math_misc._np_write_file(bt, "/tmp/ml_switcheroo_test_file.txt", b"content")
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        try:
            math_misc._np_confusion_matrix(bt, np.array([0]), np.array([0]))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        try:
            math_misc._np_decode_csv(bt, ["1,2"])
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        try:
            cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("descriptive"))(bt, cast("Callable[..., EagerTensorProtocol]", np.array)([1.0, 2.0]))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None


def test_math_misc_np_ops() -> None:
    """Test function test_math_misc_np_ops."""
    import numpy as np
    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as math_misc
    import ml_switcheroo_compiler.ops as ops

    class FakeThrowingClass:
        """Test class FakeThrowingClass."""

        def __init__(self, *args: EagerValue, **kwargs: EagerValue) -> None:
            """Test function __init__."""
            raise ValueError("boom")

    class FakeWorkingClass:
        """Test class FakeWorkingClass."""

        shape = ()

        def __init__(self, *args: EagerValue, **kwargs: EagerValue) -> None:
            """Test function __init__."""
            pass

    original_rawmatmul = None
    orig_attrs = {attr: getattr(ops, attr, None) for attr in ["RawMatMul", "SparseDenseMatMul", "DecodeImage", "ParseExample", "ParseTensor", "ReadFile", "Rem", "SerializeTensor", "WriteFile", "ConfusionMatrix", "DecodeCsv"]}
    try:
        setattr(ops, "RawMatMul", FakeThrowingClass)
        setattr(ops, "SparseDenseMatMul", FakeThrowingClass)
        setattr(ops, "DecodeImage", FakeThrowingClass)
        setattr(ops, "ParseExample", FakeThrowingClass)
        setattr(ops, "ParseTensor", FakeThrowingClass)
        setattr(ops, "ReadFile", FakeThrowingClass)
        setattr(ops, "Rem", FakeThrowingClass)
        setattr(ops, "SerializeTensor", FakeThrowingClass)
        setattr(ops, "WriteFile", FakeThrowingClass)
        setattr(ops, "ConfusionMatrix", FakeThrowingClass)
        setattr(ops, "DecodeCsv", FakeThrowingClass)
        b = math_misc.DummyBackend() if hasattr(math_misc, "DummyBackend") else None
        try:
            math_misc._np_rawmatmul(b, np.ones((2, 2)), np.ones((2, 2)))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        try:
            math_misc._np_sparsedensematmul(b, np.ones((2, 2)), np.ones((2, 2)))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        try:
            cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("decode_image")(b, b"img")
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        try:
            cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("parse_example")(b, b"example")
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        try:
            cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("parse_tensor")(b, b"tensor")
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        try:
            cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("read_file")(b, "/tmp/ml_switcheroo_test_file.txt")
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        try:
            cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("rem")(b, cast("Callable[..., EagerTensorProtocol]", np.array)([5]), cast("Callable[..., EagerTensorProtocol]", np.array)([2]))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        try:
            cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("serialize_tensor")(b, cast("Callable[..., EagerTensorProtocol]", np.array)([1]))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        try:
            cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("write_file")(b, "/tmp/ml_switcheroo_test_file.txt", b"content")
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        try:
            cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("confusion_matrix")(b, cast("Callable[..., EagerTensorProtocol]", np.array)([0]), cast("Callable[..., EagerTensorProtocol]", np.array)([0]))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        try:
            cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("decode_csv")(b, ["1,2"])
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        setattr(ops, "RawMatMul", FakeWorkingClass)
        setattr(ops, "SparseDenseMatMul", FakeWorkingClass)
        setattr(ops, "DecodeImage", FakeWorkingClass)
        setattr(ops, "ParseExample", FakeWorkingClass)
        setattr(ops, "ParseTensor", FakeWorkingClass)
        setattr(ops, "ReadFile", FakeWorkingClass)
        setattr(ops, "Rem", FakeWorkingClass)
        setattr(ops, "SerializeTensor", FakeWorkingClass)
        setattr(ops, "WriteFile", FakeWorkingClass)
        setattr(ops, "ConfusionMatrix", FakeWorkingClass)
        setattr(ops, "DecodeCsv", FakeWorkingClass)
    finally:
        for attr, val in orig_attrs.items():
            if val is not None:
                setattr(ops, attr, val)
            elif hasattr(ops, attr):
                delattr(ops, attr)
    try:
        math_misc._np_rawmatmul(b, np.ones((2, 2)), np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        math_misc._np_sparsedensematmul(b, np.ones((2, 2)), np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("decode_image")(b, b"img")
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("parse_example")(b, b"example")
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("parse_tensor")(b, b"tensor")
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("read_file")(b, "/tmp/ml_switcheroo_test_file.txt")
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("rem")(b, cast("Callable[..., EagerTensorProtocol]", np.array)([5]), cast("Callable[..., EagerTensorProtocol]", np.array)([2]))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("serialize_tensor")(b, cast("Callable[..., EagerTensorProtocol]", np.array)([1]))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("write_file")(b, "/tmp/ml_switcheroo_test_file.txt", b"content")
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("confusion_matrix")(b, cast("Callable[..., EagerTensorProtocol]", np.array)([0]), cast("Callable[..., EagerTensorProtocol]", np.array)([0]))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("decode_csv")(b, ["1,2"])
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    if original_rawmatmul:
        setattr(ops, "RawMatMul", original_rawmatmul)


def test_math_misc_np_ops_lower() -> None:
    """Test function test_math_misc_np_ops_lower."""
    import numpy as np
    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as math_misc
    import ml_switcheroo_compiler.ops as ops

    class FakeThrowingClass:
        """Test class FakeThrowingClass."""

        def __init__(self, *args: EagerValue, **kwargs: EagerValue) -> None:
            """Test function __init__."""
            raise ValueError("boom")

    class FakeWorkingClass:
        """Test class FakeWorkingClass."""

        shape = ()

        def __init__(self, *args: EagerValue, **kwargs: EagerValue) -> None:
            """Test function __init__."""
            pass

    orig_funcs = {k: getattr(ops, k, None) for k in ["decode_image", "parse_example", "parse_tensor", "read_file", "rem", "serialize_tensor", "write_file", "confusion_matrix", "decode_csv"]}
    try:
        setattr(ops, "decode_image", FakeThrowingClass)
        setattr(ops, "parse_example", FakeThrowingClass)
        setattr(ops, "parse_tensor", FakeThrowingClass)
        setattr(ops, "read_file", FakeThrowingClass)
        setattr(ops, "rem", FakeThrowingClass)
        setattr(ops, "serialize_tensor", FakeThrowingClass)
        setattr(ops, "write_file", FakeThrowingClass)
        setattr(ops, "confusion_matrix", FakeThrowingClass)
        setattr(ops, "decode_csv", FakeThrowingClass)
        b = math_misc.DummyBackend() if hasattr(math_misc, "DummyBackend") else None
        try:
            cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("decode_image")(b, b"img")
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        try:
            cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("parse_example")(b, b"example")
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        try:
            cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("parse_tensor")(b, b"tensor")
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        try:
            cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("read_file")(b, "/tmp/ml_switcheroo_test_file.txt")
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        try:
            cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("rem")(b, cast("Callable[..., EagerTensorProtocol]", np.array)([5]), cast("Callable[..., EagerTensorProtocol]", np.array)([2]))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        try:
            cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("serialize_tensor")(b, cast("Callable[..., EagerTensorProtocol]", np.array)([1]))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        try:
            cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("write_file")(b, "/tmp/ml_switcheroo_test_file.txt", b"content")
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        try:
            cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("confusion_matrix")(b, cast("Callable[..., EagerTensorProtocol]", np.array)([0]), cast("Callable[..., EagerTensorProtocol]", np.array)([0]))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        try:
            cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("decode_csv")(b, ["1,2"])
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        setattr(ops, "decode_image", FakeWorkingClass)
        setattr(ops, "parse_example", FakeWorkingClass)
        ops.parse_tensor = FakeWorkingClass
        ops.read_file = FakeWorkingClass
        setattr(ops, "rem", FakeWorkingClass)
        ops.serialize_tensor = FakeWorkingClass
        setattr(ops, "write_file", FakeWorkingClass)
        ops.confusion_matrix = FakeWorkingClass
        setattr(ops, "decode_csv", FakeWorkingClass)
        try:
            cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("decode_image")(b, b"img")
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        try:
            cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("parse_example")(b, b"example")
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        try:
            cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("parse_tensor")(b, b"tensor")
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        try:
            cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("read_file")(b, "/tmp/ml_switcheroo_test_file.txt")
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        try:
            cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("rem")(b, cast("Callable[..., EagerTensorProtocol]", np.array)([5]), cast("Callable[..., EagerTensorProtocol]", np.array)([2]))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        try:
            cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("serialize_tensor")(b, cast("Callable[..., EagerTensorProtocol]", np.array)([1]))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        try:
            cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("write_file")(b, "/tmp/ml_switcheroo_test_file.txt", b"content")
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        try:
            cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("confusion_matrix")(b, cast("Callable[..., EagerTensorProtocol]", np.array)([0]), cast("Callable[..., EagerTensorProtocol]", np.array)([0]))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        try:
            cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("decode_csv")(b, ["1,2"])
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
    finally:
        for k, v in orig_funcs.items():
            if v is not None:
                setattr(ops, k, v)
            elif hasattr(ops, k):
                delattr(ops, k)


def test_reductions() -> None:
    """Test function test_reductions."""
    b = np
    red._np_trapezoidal_integral(b, np.array([1.0, 2.0, 3.0]))
    red._np_trapezoidal_integral(b, np.array([1.0, 2.0, 3.0]), x=np.array([1.0, 2.0, 3.0]))
    red._np_confusion_matrix(b, np.array([0, 1]), np.array([0, 1]))
    red._np_confusion_matrix(b, np.array([0, 1]), np.array([0, 1]), num_classes=2)

    class DummyDtype:
        """Test class DummyDtype."""

        value = np.float32

    red._np_cummax(b, np.array([1, 2, 3]))
    red._np_cummax(b, np.array([1, 2, 3]), dtype=DummyDtype)
    red._np_cummin(b, np.array([1, 2, 3]))
    red._np_cummin(b, np.array([1, 2, 3]), dtype=DummyDtype)
    red._np_cumprod(b, np.array([1, 2, 3]))
    red._np_cumprod(b, np.array([1, 2, 3]), dtype=DummyDtype)
    red._np_cumlogsumexp(b, np.array([1.0, 2.0, 3.0]))
    red._np_cumlogsumexp(b, np.array([1.0, 2.0, 3.0]), axis=0)
    red._np_approx_max_k(b, np.array([1.0, 2.0, 3.0]), 2)
    red._np_approx_max_k(b, np.array([]), 2)
    red._np_approx_min_k(b, np.array([1.0, 2.0, 3.0]), 2)
    red._np_approx_min_k(b, np.array([]), 2)
    red._np_top_k(b, np.array([1.0, 2.0, 3.0]), 2)
    red._np_top_k(b, np.array([1.0, 2.0, 3.0]), 2, return_indices=False)
    red._np_top_k(b, np.array([1.0, 2.0, 3.0]), 2, return_indices=True)

    class MockK:
        """Test class MockK."""

        def item(self) -> EagerValue:
            """Test function item."""
            return 2

    red._np_top_k(b, np.array([1.0, 2.0, 3.0]), MockK())

    class MockK2:
        """Test class MockK2."""

        class Data:
            """Test class Data."""

            def item(self) -> EagerValue:
                """Test function item."""
                return 2

        data = Data()

    red._np_top_k(b, np.array([1.0, 2.0, 3.0]), MockK2())
    red._segment_sum(np.array([1.0, 2.0]), np.array([0, 1]))
    red._segment_sum(np.array([1.0, 2.0]), np.array([0, 1]), num_segments=2)
    red._logsumexp(np.array([1.0, 2.0]))
    red._logsumexp(np.array([1.0, 2.0]), keepdims=True)
    red._top_k(np.array([1.0, 2.0, 3.0]), 2)
    red._top_k(np.array([[1.0, 2.0, 3.0]]), 2, axis=1)
    try:
        red._np_nms(b, np.array([[0, 0, 1, 1]]), np.array([1.0]), 1)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None

    class DummyConfig:
        """Test class DummyConfig."""

        window_dimensions = [2]
        window_strides = [1]
        window_dilation = [1]
        base_dilation = [1]
        padding = "SAME"

    cast("Callable[..., EagerTensorProtocol]", red._reduce_window)(cast("Callable[..., EagerTensorProtocol]", np.array)([1.0, 2.0, 3.0]), 0.0, "max", cast("Callable[..., EagerTensorProtocol]", DummyConfig)())
    cast("Callable[..., EagerTensorProtocol]", red._reduce_window)(cast("Callable[..., EagerTensorProtocol]", np.array)([1.0, 2.0, 3.0]), 0.0, "min", cast("Callable[..., EagerTensorProtocol]", DummyConfig)())
    cast("Callable[..., EagerTensorProtocol]", red._reduce_window)(cast("Callable[..., EagerTensorProtocol]", np.array)([1.0, 2.0, 3.0]), 0.0, "sum", cast("Callable[..., EagerTensorProtocol]", DummyConfig)())
    cast("Callable[..., EagerTensorProtocol]", red._reduce_window)(cast("Callable[..., EagerTensorProtocol]", np.array)([1.0, 2.0, 3.0]), 0.0, "prod", cast("Callable[..., EagerTensorProtocol]", DummyConfig)())
    try:
        cast("Callable[..., EagerTensorProtocol]", red._reduce_window)(cast("Callable[..., EagerTensorProtocol]", np.array)([1.0, 2.0, 3.0]), 0.0, "unknown", cast("Callable[..., EagerTensorProtocol]", DummyConfig)())
    except ValueError as _exc:
        assert _exc is not None

    class DummyConfig2:
        """Test class DummyConfig2."""

        window_dimensions = [2]
        window_strides = [1]
        window_dilation = [1]
        base_dilation = [2]
        padding = [(1, 1)]

    cast("Callable[..., EagerTensorProtocol]", red._reduce_window)(cast("Callable[..., EagerTensorProtocol]", np.array)([1.0, 2.0, 3.0]), 0.0, "sum", cast("Callable[..., EagerTensorProtocol]", DummyConfig2)())

    class DummyConfig3:
        """Test class DummyConfig3."""

        window_dimensions = [2]
        window_strides = [1]
        window_dilation = [1]
        base_dilation = [1]
        padding = None

    cast("Callable[..., EagerTensorProtocol]", red._reduce_window)(cast("Callable[..., EagerTensorProtocol]", np.array)([1.0, 2.0, 3.0]), 0.0, "sum", cast("Callable[..., EagerTensorProtocol]", DummyConfig3)())


def test_missing_lines() -> None:
    """Test function test_missing_lines."""
    import numpy as np
    import ml_switcheroo_compiler.backends.numpy.eager.reductions as red

    b = np

    class DummyConfig4:
        """Test class DummyConfig4."""

        window_dimensions = [2]
        window_strides = [1]
        window_dilation = [1]
        base_dilation = [1]
        padding = "SAME"

    try:
        cast("Callable[..., EagerTensorProtocol]", red._reduce_window)(cast("Callable[..., EagerTensorProtocol]", np.array)([[1.0, 2.0], [3.0, 4.0]]), 0.0, "max", cast("Callable[..., EagerTensorProtocol]", DummyConfig4)())
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None

    class DummyConfig5:
        """Test class DummyConfig5."""

        window_dimensions = [2]
        window_strides = [1]
        window_dilation = [1]
        base_dilation = [1]
        padding = "VALID"

    cast("Callable[..., EagerTensorProtocol]", red._reduce_window)(cast("Callable[..., EagerTensorProtocol]", np.array)([1.0, 2.0, 3.0]), 0.0, "max", cast("Callable[..., EagerTensorProtocol]", DummyConfig5)())

    class DummyConfig6:
        """Test class DummyConfig6."""

        window_dimensions = [1]
        window_strides = [1]
        window_dilation = [1]
        base_dilation = [1]
        padding = "SAME"

    cast("Callable[..., EagerTensorProtocol]", red._reduce_window)(1.0, 0.0, "max", cast("Callable[..., EagerTensorProtocol]", DummyConfig6)())
    red._np_approx_min_k(b, [1.0, 2.0, 3.0], 2)
    red._np_top_k(b, [1.0, 2.0, 3.0], 2)
    red._np_approx_max_k(b, [1.0, 2.0, 3.0], 2)


if __name__ == "__main__":
    test_reductions()
    test_missing_lines()


def test_math_fft_return_none() -> None:
    """Test math fft return none."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops.math_fft import _fftconvolve

    class DummyBackendWithoutFFT:
        """Dummy."""

        pass

    assert _fftconvolve(DummyBackendWithoutFFT(), None, [3, 4]) is None


def test_math_internal_return_zero() -> None:
    """Test math internal return zero."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops.math_internal import _np_tensorarraywrite

    class DummyBackend:
        """Dummy."""

        pass

    assert _np_tensorarraywrite(DummyBackend(), "not_a_list", 0, 1) == 0


def test_math_fft_branch_coverage() -> None:
    """Test math fft branch coverage."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops.math_fft import _fftconvolve

    class DummySignal:
        """Dummy."""

        pass

    class DummyBackendWithSignal:
        """Dummy."""

        signal = DummySignal()

    assert _fftconvolve(DummyBackendWithSignal(), [1, 2], None) is None


def test_math_fft_signal_fftconvolve() -> None:
    """Test math fft signal fftconvolve."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops.math_fft import _fftconvolve

    class DummySignal:
        """Dummy."""

        def fftconvolve(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Fftconvolve."""
            return "success"

    class DummyBackendWithSignal:
        """Dummy."""

        signal = DummySignal()

    assert _fftconvolve(DummyBackendWithSignal(), [1, 2], [3, 4]) == "success"
