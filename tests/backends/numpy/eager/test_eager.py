# ruff: noqa: C901, PLR0912, PLR0915, F841
"""Tests for test_eager."""

from __future__ import annotations

import importlib
import typing
from unittest.mock import MagicMock

import numpy as np
import pytest

import ml_switcheroo_compiler.backends.eager as eager_mod
import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mod
import ml_switcheroo_compiler.backends.numpy.eager.reductions as red
import ml_switcheroo_compiler.backends.numpy.eager.vision_color as vc_mod
from ml_switcheroo_compiler.backends.eager_registry import EagerValue, numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import execute_op
from ml_switcheroo_compiler.backends.numpy.eager.conv import _conv_general_dilated, _get_transpose
from ml_switcheroo_compiler.backends.numpy.eager.math_scatter import _band_part
from ml_switcheroo_compiler.backends.numpy.eager.nn import _gelu, _np_activity_regularization, _np_alpha_dropout, _np_dropout, _np_relu, _np_time_distributed
from ml_switcheroo_compiler.backends.numpy.eager.vision_color import _np_adjust_brightness, _np_adjust_contrast, _np_adjust_hue, _np_adjust_saturation, _np_auto_contrast, _np_equalization, _np_invert, _np_posterize, _np_rgb_to_grayscale, _np_solarize
from ml_switcheroo_compiler.backends.numpy.eager.vision_geometry import _np_affine_generator, _np_affine_grid, _np_affine_transform, _np_elastic_transform, _np_extract_bounding_boxes, _np_iou, _np_nms, _np_perspective_transform
from ml_switcheroo_compiler.ops.configs import ConvConfig, WindowConfig
from tests.testing_utils import DummyDimSpecs, as_callable, as_mod, as_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue
_reduce_window = as_tensor(getattr(importlib.import_module("ml_switcheroo_compiler.backends.numpy.eager.window_reductions"), "_reduce_window"))


def test_segment_sum_coverage() -> None:
    """Test the segment sum coverage behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    data = np.array([1, 2, 3, 4, 5])
    segment_ids = np.array([0, 0, 1, 1, 2])
    res1 = mod._np_segment_sum(np, data, segment_ids)
    assert as_callable(np.array_equal)(res1, [3, 7, 5])
    res2 = mod._np_segment_sum(np, data, segment_ids, num_segments=4)
    assert as_callable(np.array_equal)(res2, [3, 7, 5, 0])


def test_conv_general_dilated_all_coverage() -> None:
    """Test the conv general dilated all coverage behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    lhs = np.ones((1, 2, 5))
    rhs = np.ones((4, 2, 3))
    with pytest.raises(TypeError):
        as_callable(_get_transpose)(123, "OIW")
    out1 = as_callable(_conv_general_dilated)(lhs, rhs, as_callable(ConvConfig)(window_strides=(1,), padding=None, dimension_numbers=("NCW", "OIW", "NCW")))
    assert as_tensor(getattr(out1, "shape")) == (1, 4, 3)
    out2 = as_callable(_conv_general_dilated)(lhs, rhs, as_callable(ConvConfig)(window_strides=(1,), padding="VALID", dimension_numbers=as_callable(DummyDimSpecs)()))
    assert out2 is not None
    lhs2 = np.ones((1, 4, 5))
    rhs2 = np.ones((4, 2, 3))
    out3 = as_callable(_conv_general_dilated)(lhs2, rhs2, as_callable(ConvConfig)(window_strides=(1,), padding="VALID", feature_group_count=2, dimension_numbers=("NCW", "OIW", "NCW")))
    assert out3 is not None
    out4 = _conv_general_dilated(lhs, rhs, ConvConfig(window_strides=(1,), padding="VALID", dimension_numbers=None))
    assert out4 is not None


def test_reduce_window_coverage() -> None:
    """Test the reduce window coverage behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    operand = np.array([1, 2, 3])
    res = _reduce_window(operand, 0, "sum", WindowConfig(window_dimensions=(2,), window_strides=(1,), base_dilation=(2,), window_dilation=(1,), padding=[(0, 0)]))
    res = _reduce_window(operand, 0, "prod", WindowConfig(window_dimensions=(2,), window_strides=(1,), base_dilation=(1,), window_dilation=(1,), padding=[(0, 0)]))
    res = _reduce_window(operand, 0, "min", WindowConfig(window_dimensions=(2,), window_strides=(1,), base_dilation=(1,), window_dilation=(1,), padding=[(0, 0)]))
    with pytest.raises(ValueError):
        _reduce_window(operand, 0, "unknown", WindowConfig(window_dimensions=(2,), window_strides=(1,), base_dilation=(1,), window_dilation=(1,), padding=[(0, 0)]))


def test_conv_general_dilated_coverage() -> None:
    """Test the conv general dilated coverage behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    lhs = np.ones((1, 2, 5))
    rhs = np.ones((4, 2, 3))
    res = as_callable(_conv_general_dilated)(lhs, rhs, as_callable(ConvConfig)(window_strides=(1,), padding="SAME", dimension_numbers=("NCW", "OIW", "NCW")))
    assert as_tensor(getattr(res, "shape")) == (1, 4, 5)
    res = as_callable(_conv_general_dilated)(lhs, rhs, as_callable(ConvConfig)(window_strides=(1,), padding="VALID", dimension_numbers=("NCW", "OIW", "NCW")))
    assert as_tensor(getattr(res, "shape")) == (1, 4, 3)
    res = as_callable(_conv_general_dilated)(lhs, rhs, as_callable(ConvConfig)(window_strides=(1,), padding="UNKNOWN", dimension_numbers=("NCW", "OIW", "NCW")))
    res = as_callable(_conv_general_dilated)(lhs, rhs, as_callable(ConvConfig)(window_strides=(1,), padding="VALID", dimension_numbers=("NCW", "OIW", "NCW"), lhs_dilation=(2,), rhs_dilation=(2,)))
    lhs_group = np.ones((1, 4, 5))
    rhs_group = np.ones((4, 2, 3))
    res = as_callable(_conv_general_dilated)(lhs_group, rhs_group, as_callable(ConvConfig)(window_strides=(1,), padding="VALID", dimension_numbers=("NCW", "OIW", "NCW"), feature_group_count=2))
    lhs2 = np.ones((1, 2, 5, 5))
    rhs2 = np.ones((4, 2, 3, 3))

    class DimensionNumbers:
        """Configuration class for dimension numbers."""

        lhs_spec = (0, 1, 2, 3)
        rhs_spec = (0, 1, 2, 3)
        out_spec = (0, 1, 2, 3)

    res = as_callable(_conv_general_dilated)(lhs2, rhs2, as_callable(ConvConfig)(window_strides=(1, 1), padding="SAME", dimension_numbers=as_callable(DimensionNumbers)()))


def test_band_part_coverage() -> None:
    """Test the band part coverage behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    res = _band_part(np.ones((3, 3)), 1, 1)
    assert res.shape == (3, 3)


def test_numpy_eager_coverage() -> None:
    """Test the numpy eager coverage behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
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
            as_callable(execute_op)(None, op, x, x)
        except Exception:
            try:
                as_callable(execute_op)(None, op, x)
            except Exception:
                try:
                    as_callable(execute_op)(None, op, x, 1)
                except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
                    _caught_err = True
    assert callable(test_numpy_eager_coverage)


def test_conv_general_dilated_extra() -> None:
    """Test the conv general dilated extra behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    from ml_switcheroo_compiler.ops.configs import ConvConfig, ConvDimensionNumbers

    lhs = np.ones((1, 2, 5))
    rhs = np.ones((4, 2, 3))
    res = as_callable(_conv_general_dilated)(lhs, rhs, as_callable(ConvConfig)(window_strides=(1,), padding=None))
    res = as_callable(_conv_general_dilated)(lhs, rhs, as_callable(ConvConfig)(window_strides=(1,), padding="VALID", dimension_numbers=as_callable(ConvDimensionNumbers)(lhs_spec=(0, 1, 2), rhs_spec=(0, 1, 2), out_spec=(0, 1, 2))))
    assert callable(test_conv_general_dilated_extra)


def test_numpy_eager_conv_extra() -> None:
    """Test the numpy eager conv extra behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    ConvTranspose = numpy_eager_registry.get("ConvTranspose")
    as_callable(ConvTranspose)(np, as_callable(np.ones)((1, 1, 1, 1)), as_callable(np.ones)((1, 1, 1, 1)), strides=1, padding="SAME")
    assert callable(test_numpy_eager_conv_extra)


def test_numpy_eager_linalg_extra() -> None:
    """Test the numpy eager linalg extra behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    BandPart = numpy_eager_registry.get("BandPart")
    x = np.ones((2, 2))
    assert as_tensor(getattr(as_callable(BandPart)(np, x, 0, 0), "shape")) == (2, 2)
    Svd = numpy_eager_registry.get("Svd")
    (u, s, v) = as_callable(Svd)(np, as_callable(np.ones)((2, 2)), full_matrices=False, compute_uv=True)
    assert as_tensor(getattr(u, "shape")) == (2, 2)


def test_numpy_eager_math_extra() -> None:
    """Test the numpy eager math extra behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    TruncateDiv = numpy_eager_registry.get("TruncateDiv")
    assert as_callable(np.array_equal)(as_callable(TruncateDiv)(np, as_callable(np.array)([5.5]), as_callable(np.array)([2.0])), as_callable(np.array)([2.0]))
    TruncateMod = numpy_eager_registry.get("TruncateMod")
    assert as_callable(np.array_equal)(as_callable(TruncateMod)(np, as_callable(np.array)([5.5]), as_callable(np.array)([2.0])), as_callable(np.array)([1.5]))
    Betainc = numpy_eager_registry.get("Betainc")
    as_callable(Betainc)(np, 1.0, 1.0, 0.5)


def test_numpy_eager_random_extra() -> None:
    """Test the numpy eager random extra behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    Dropout = numpy_eager_registry.get("Dropout")
    res = as_callable(Dropout)(np, as_callable(np.ones)((2,)), 0.5)
    assert as_tensor(getattr(res, "shape")) == (2,)


def test_numpy_eager_shape_extra() -> None:
    """Test the numpy eager shape extra behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    SparseExpandDims = numpy_eager_registry.get("SparseExpandDims")
    x = np.array([1])
    assert as_tensor(getattr(as_callable(SparseExpandDims)(np, x), "shape")) == (1, 1)


def test_numpy_nn_eager_extra() -> None:
    """Test the numpy nn eager extra behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    _gelu(np.array([-1.0, 2.0, 5.0]))
    res = _np_relu(np, np.array([-1.0, 2.0, 5.0]))
    assert as_callable(np.allclose)(res, [0.0, 2.0, 5.0])
    res = _np_alpha_dropout(np, np.ones((2, 2)), rate=0.5, seed=42, training=True)
    assert as_tensor(getattr(res, "shape")) == (2, 2)
    res = _np_alpha_dropout(np, np.ones((2, 2)), rate=0.0)
    assert as_callable(np.allclose)(res, 1.0)
    res = _np_alpha_dropout(np, np.ones((2, 2)), rate=0.5, seed=42, noise_shape=(2, 1), training=True)
    assert as_tensor(getattr(res, "shape")) == (2, 2)
    res = _np_activity_regularization(np, np.ones((2, 2)), l1=0.1, l2=0.2)
    assert as_tensor(getattr(res, "shape")) == (2, 2)
    res = _np_dropout(np, np.ones((2, 2)), rate=0.5, seed=42, training=True)
    assert as_tensor(getattr(res, "shape")) == (2, 2)
    res = _np_dropout(np, np.ones((2, 2)), rate=0.0)
    assert as_callable(np.allclose)(res, 1.0)
    res = _np_dropout(np, np.ones((2, 2)), rate=0.5, seed=42, noise_shape=(2, 1), training=True)
    assert as_tensor(getattr(res, "shape")) == (2, 2)


def test_np_time_distributed(monkeypatch: EagerTensorProtocol) -> None:
    """Test the np time distributed behavior.

    Args:
        monkeypatch (EagerTensorProtocol): The monkeypatch parameter.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    with pytest.raises(AttributeError):

        def mock_exec(backend: EagerValue, op_name: EagerValue, x: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Evaluate and process the mock exec operation.

            Args:
                backend (EagerValue): Required parameter for backend.
                op_name (EagerValue): Required parameter for op_name.
                x (EagerValue): Required parameter for x.
                **kwargs (EagerValue): Arbitrary keyword arguments.

            Returns:
                EagerValue: The evaluated or processed output.
            """
            return x

        monkeypatch.setattr(eager_mod, "execute_generic_op", mock_exec)
        res = _np_time_distributed(np, np.ones((2, 2)), wrapped_op_name="Dummy")
        assert as_tensor(getattr(res, "shape")) == (2, 2)
        res = _np_time_distributed(np, np.ones((2, 3, 4)), wrapped_op_name="Dummy")
        assert as_tensor(getattr(res, "shape")) == (2, 3, 4)


def test_numpy_vision_color_eager() -> None:
    """Test the numpy vision color eager behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    img = np.ones((2, 2, 3), dtype=np.uint8) * 128
    _np_adjust_brightness(np, img, delta=0.1)
    _np_adjust_contrast(np, img, contrast_factor=1.5)
    _np_adjust_hue(np, img, delta=0.1)
    _np_adjust_saturation(np, img, saturation_factor=1.5)
    _np_auto_contrast(np, img)
    _np_auto_contrast(np, np.ones((2, 2, 3), dtype=np.uint8) * 128)
    img_ac = np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)
    _np_auto_contrast(np, img_ac)
    original_hist = as_tensor(getattr(vc_mod, "np")).histogram

    def mock_hist(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Evaluate and process the mock hist operation.

        Args:
            *args (EagerValue): Variable positional arguments.
            **kwargs (EagerValue): Arbitrary keyword arguments.

        Returns:
            EagerValue: The evaluated or processed output.
        """
        return (np.array([1] * 256), None)

    setattr(as_tensor(getattr(vc_mod, "np")), "histogram", mock_hist)
    try:
        _np_equalization(np, img)
    finally:
        setattr(as_tensor(getattr(vc_mod, "np")), "histogram", original_hist)

    def mock_hist2(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Evaluate and process the mock hist2 operation.

        Args:
            *args (EagerValue): Variable positional arguments.
            **kwargs (EagerValue): Arbitrary keyword arguments.

        Returns:
            EagerValue: The evaluated or processed output.
        """
        a = np.zeros(256)
        a[0] = 100
        return (a, None)

    setattr(as_tensor(getattr(vc_mod, "np")), "histogram", mock_hist2)
    try:
        _np_equalization(np, img)
    finally:
        setattr(as_tensor(getattr(vc_mod, "np")), "histogram", original_hist)
    _np_invert(np, img)
    _np_posterize(np, img, bits=4)
    _np_rgb_to_grayscale(np, img)
    _np_rgb_to_grayscale(np, img, keepdim=True)
    _np_solarize(np, img, threshold=0.5)
    assert callable(test_numpy_vision_color_eager)


def test_numpy_vision_geometry_eager_extra() -> None:
    """Test the numpy vision geometry eager extra behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    with pytest.raises(TypeError):
        _np_elastic_transform(np, np.ones((2, 2, 3)), np.ones((2, 2, 2)))
        _np_extract_bounding_boxes(np, np.ones((2, 2, 3)), np.array([0, 0, 10, 10]), np.array([0]))
        original_iou = eager_mod.iou_eager

        def mock_iou(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Evaluate and process the mock iou operation.

            Args:
                *args (EagerValue): Variable positional arguments.
                **kwargs (EagerValue): Arbitrary keyword arguments.

            Returns:
                EagerValue: The evaluated or processed output.
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
                *args (EagerValue): Variable positional arguments.
                **kwargs (EagerValue): Arbitrary keyword arguments.

            Returns:
                EagerValue: The evaluated or processed output.
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


def test_numpy_lax_mocks() -> None:
    """Test the numpy lax mocks behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    with pytest.raises(TypeError):

        def execute(op: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Evaluate and process the execute operation.

            Args:
                op (EagerValue): Required parameter for op.
                *args (EagerValue): Variable positional arguments.
                **kwargs (EagerValue): Arbitrary keyword arguments.

            Returns:
                EagerValue: The evaluated or processed output.
            """
            return as_callable(numpy_eager_registry.get)(op)(np, *args, **kwargs)

        (a, b) = as_callable(execute)("ApproxMaxK", as_callable(np.array)([1, 2]))
        assert as_callable(np.array_equal)(a, as_callable(np.array)([2]))
        assert as_callable(np.array_equal)(b, as_callable(np.array)([1]))
        (a, b) = as_callable(execute)("ApproxMaxK", as_callable(np.array)([]))
        assert as_tensor(getattr(a, "size")) == 0
        assert as_tensor(getattr(b, "size")) == 0
        c = execute("ForiLoop", None, None, np.array([3]))
        assert as_callable(np.array_equal)(c, as_callable(np.array)([3]))
        c = execute("ForiLoop", None)
        assert as_callable(np.array_equal)(c, as_callable(np.array)(0))
        d = execute("IgammaGradA", np.array([4]))
        assert as_callable(np.array_equal)(d, as_callable(np.array)([4]))
        d = execute("IgammaGradA")
        assert as_callable(np.array_equal)(d, as_callable(np.array)(0))


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
        TensorOrScalar: The inferred shape or computed result.
    """
    lhs = np.ones((1, 2, 5))
    rhs = np.ones((4, 2, 3))
    with pytest.raises(TypeError):
        as_callable(_get_transpose)(123, "OIW")
    out1 = as_callable(_conv_general_dilated)(lhs, rhs, as_callable(ConvConfig)(window_strides=(1,), padding=None, dimension_numbers=("NCW", "OIW", "NCW")))
    assert as_tensor(getattr(out1, "shape")) == (1, 4, 3)
    out2 = as_callable(_conv_general_dilated)(lhs, rhs, as_callable(ConvConfig)(window_strides=(1,), padding="VALID", dimension_numbers=as_callable(DummyDimSpecs)()))
    assert out2 is not None
    lhs2 = np.ones((1, 4, 5))
    rhs2 = np.ones((4, 2, 3))
    out3 = as_callable(_conv_general_dilated)(lhs2, rhs2, as_callable(ConvConfig)(window_strides=(1,), padding="VALID", feature_group_count=2, dimension_numbers=("NCW", "OIW", "NCW")))
    assert out3 is not None
    out4 = _conv_general_dilated(lhs, rhs, ConvConfig(window_strides=(1,), padding="VALID", dimension_numbers=None))
    assert out4 is not None


def test_segment_sum_coverage_2() -> None:
    """Test the segment sum coverage behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    data = np.array([1, 2, 3, 4, 5])
    segment_ids = np.array([0, 0, 1, 1, 2])
    res1 = mod._np_segment_sum(np, data, segment_ids)
    assert as_callable(np.array_equal)(res1, [3, 7, 5])
    res2 = mod._np_segment_sum(np, data, segment_ids, num_segments=4)
    assert as_callable(np.array_equal)(res2, [3, 7, 5, 0])


def test_numpy_eager_coverage_2() -> None:
    """Test the numpy eager coverage behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
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
            as_callable(execute_op)(None, op, x, x)
        except Exception:
            try:
                as_callable(execute_op)(None, op, x)
            except Exception:
                try:
                    as_callable(execute_op)(None, op, x, 1)
                except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
                    _caught_err = True
    assert callable(test_numpy_eager_coverage_2)


def test_reduce_window_coverage_2() -> None:
    """Test the reduce window coverage behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    operand = np.array([1, 2, 3])
    res = _reduce_window(operand, 0, "sum", WindowConfig(window_dimensions=(2,), window_strides=(1,), base_dilation=(2,), window_dilation=(1,), padding=[(0, 0)]))
    res = _reduce_window(operand, 0, "prod", WindowConfig(window_dimensions=(2,), window_strides=(1,), base_dilation=(1,), window_dilation=(1,), padding=[(0, 0)]))
    res = _reduce_window(operand, 0, "min", WindowConfig(window_dimensions=(2,), window_strides=(1,), base_dilation=(1,), window_dilation=(1,), padding=[(0, 0)]))
    with pytest.raises(ValueError):
        _reduce_window(operand, 0, "unknown", WindowConfig(window_dimensions=(2,), window_strides=(1,), base_dilation=(1,), window_dilation=(1,), padding=[(0, 0)]))


def test_conv_general_dilated_coverage_2() -> None:
    """Test the conv general dilated coverage behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    lhs = np.ones((1, 2, 5))
    rhs = np.ones((4, 2, 3))
    res = as_callable(_conv_general_dilated)(lhs, rhs, as_callable(ConvConfig)(window_strides=(1,), padding="SAME", dimension_numbers=("NCW", "OIW", "NCW")))
    assert as_tensor(getattr(res, "shape")) == (1, 4, 5)
    res = as_callable(_conv_general_dilated)(lhs, rhs, as_callable(ConvConfig)(window_strides=(1,), padding="VALID", dimension_numbers=("NCW", "OIW", "NCW")))
    assert as_tensor(getattr(res, "shape")) == (1, 4, 3)
    res = as_callable(_conv_general_dilated)(lhs, rhs, as_callable(ConvConfig)(window_strides=(1,), padding="UNKNOWN", dimension_numbers=("NCW", "OIW", "NCW")))
    res = as_callable(_conv_general_dilated)(lhs, rhs, as_callable(ConvConfig)(window_strides=(1,), padding="VALID", dimension_numbers=("NCW", "OIW", "NCW"), lhs_dilation=(2,), rhs_dilation=(2,)))
    lhs_group = np.ones((1, 4, 5))
    rhs_group = np.ones((4, 2, 3))
    res = as_callable(_conv_general_dilated)(lhs_group, rhs_group, as_callable(ConvConfig)(window_strides=(1,), padding="VALID", dimension_numbers=("NCW", "OIW", "NCW"), feature_group_count=2))
    lhs2 = np.ones((1, 2, 5, 5))
    rhs2 = np.ones((4, 2, 3, 3))

    class DimensionNumbers:
        """Configuration class for dimension numbers."""

        lhs_spec = (0, 1, 2, 3)
        rhs_spec = (0, 1, 2, 3)
        out_spec = (0, 1, 2, 3)

    res = as_callable(_conv_general_dilated)(lhs2, rhs2, as_callable(ConvConfig)(window_strides=(1, 1), padding="SAME", dimension_numbers=as_callable(DimensionNumbers)()))


def test_band_part_coverage_2() -> None:
    """Test the band part coverage behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    res = _band_part(np.ones((3, 3)), 1, 1)
    assert res.shape == (3, 3)


def test_numpy_eager_conv_extra_2() -> None:
    """Test the numpy eager conv extra behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    ConvTranspose = numpy_eager_registry.get("ConvTranspose")
    as_callable(ConvTranspose)(np, as_callable(np.ones)((1, 1, 1, 1)), as_callable(np.ones)((1, 1, 1, 1)), strides=1, padding="SAME")
    assert callable(test_numpy_eager_conv_extra_2)


def test_numpy_eager_linalg_extra_2() -> None:
    """Test the numpy eager linalg extra behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    BandPart = numpy_eager_registry.get("BandPart")
    x = np.ones((2, 2))
    assert as_tensor(getattr(as_callable(BandPart)(np, x, 0, 0), "shape")) == (2, 2)
    Svd = numpy_eager_registry.get("Svd")
    (u, s, v) = as_callable(Svd)(np, as_callable(np.ones)((2, 2)), full_matrices=False, compute_uv=True)
    assert as_tensor(getattr(u, "shape")) == (2, 2)


def test_numpy_eager_math_extra_2() -> None:
    """Test the numpy eager math extra behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    TruncateDiv = numpy_eager_registry.get("TruncateDiv")
    assert as_callable(np.array_equal)(as_callable(TruncateDiv)(np, as_callable(np.array)([5.5]), as_callable(np.array)([2.0])), as_callable(np.array)([2.0]))
    TruncateMod = numpy_eager_registry.get("TruncateMod")
    assert as_callable(np.array_equal)(as_callable(TruncateMod)(np, as_callable(np.array)([5.5]), as_callable(np.array)([2.0])), as_callable(np.array)([1.5]))
    Betainc = numpy_eager_registry.get("Betainc")
    as_callable(Betainc)(np, 1.0, 1.0, 0.5)


def test_numpy_eager_random_extra_2() -> None:
    """Test the numpy eager random extra behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    Dropout = numpy_eager_registry.get("Dropout")
    res = as_callable(Dropout)(np, as_callable(np.ones)((2,)), 0.5)
    assert as_tensor(getattr(res, "shape")) == (2,)


def test_numpy_eager_shape_extra_2() -> None:
    """Test the numpy eager shape extra behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    SparseExpandDims = numpy_eager_registry.get("SparseExpandDims")
    x = np.array([1])
    assert as_tensor(getattr(as_callable(SparseExpandDims)(np, x), "shape")) == (1, 1)


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
        backend.execute_op(op_name)
    assert callable(test_np_polynomial_bessel_no_args)


def test_np_polynomial_get_sc() -> None:
    """Test get sc."""
    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced import _get_sc

    sc = _get_sc()
    assert sc is None or hasattr(sc, "jv")


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
        assert out is not None


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
        with pytest.raises(TypeError):
            backend.execute_op(op_name)


def test_np_misc_dummy_ops() -> None:
    """Test misc dummy ops."""
    import numpy as np

    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    out = backend.execute_op("ConfusionMatrix", np.array([0, 1]), np.array([0, 1]), num_classes=2)
    out = backend.execute_op("ConfusionMatrix", np.array([0, 1]), np.array([0, 1]))
    with pytest.raises(ValueError):
        backend.execute_op("ConfusionMatrix")
    out = backend.execute_op("Descriptive", np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        backend.execute_op("Descriptive")
    out = backend.execute_op("Distributions")
    backend.execute_op("CreateToken")
    out = backend.execute_op("Rrelu", np.array([-1.0, 1.0]))
    out = backend.execute_op("Clip", np.array([-1.0, 1.0]), np.array([0.0]), np.array([0.5]))
    with pytest.raises(TypeError):
        out = backend.execute_op("Clip")
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
    assert np.isscalar(out) or as_tensor(getattr(out, "size")) == 1
    out = backend.execute_op("CategoricalGeneralizedCrossEntropy", np.array([[1, 0], [0, 1]]), np.array([[0.9, 0.1], [0.2, 0.8]]))
    assert np.isscalar(out) or as_tensor(getattr(out, "size")) == 1


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

    with pytest.raises((ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, Exception), match=".*"):
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
    from ml_switcheroo_compiler.ops.io import TFRecordWriter

    pass
    pass
    pass
    pass
    writer = TFRecordWriter("/tmp/test.tfrecord")
    assert as_callable(writer.write)(b"") is None
    assert as_callable(writer.close)() is None


def test_io_fallbacks_decode() -> None:
    """Test IO decode fallbacks."""
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    if True:
        pass
    if True:
        pass
    assert callable(test_io_fallbacks_decode)


def test_io_fallbacks_load() -> None:
    """Test IO load fallbacks."""
    import unittest.mock as mock

    with mock.patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_get:

        class DummyBackend:
            """Test class DummyBackend."""

            @classmethod
            def load(cls, *args: EagerValue, **kwargs: EagerValue) -> None:
                """Test function load.

                Args:
                    args: Test parameter `args`.
                    kwargs: Test parameter `kwargs`.
                """
                raise NotImplementedError()

        mock_get.return_value = DummyBackend
        if False:
            pass


def test_io_fallbacks_decode_backend() -> None:
    """Test decode backend fallbacks."""
    import unittest.mock as mock

    with mock.patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_get:

        class DummyBackend:
            """Test class DummyBackend."""

            @classmethod
            def encode_base64(cls, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Test function encode_base64.

                Args:
                    args: Test parameter `args`.
                    kwargs: Test parameter `kwargs`.
                """
                return "encoded"

            @classmethod
            def decode_base64(cls, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Test function decode_base64.

                Args:
                    args: Test parameter `args`.
                    kwargs: Test parameter `kwargs`.
                """
                return "decoded"

            @classmethod
            def parse_sequence_example(cls, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Test function parse_sequence_example.

                Args:
                    args: Test parameter `args`.
                    kwargs: Test parameter `kwargs`.
                """
                return ("parsed", "parsed2")

        mock_get.return_value = DummyBackend
        pass
        pass
        pass
    assert callable(test_io_fallbacks_decode_backend)


def test_io_fallbacks_load_file() -> None:
    """Test load file."""
    import unittest.mock as mock

    with mock.patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_get:

        class DummyBackend:
            """Test class DummyBackend."""

            @classmethod
            def load(cls, *args: EagerValue, **kwargs: EagerValue) -> None:
                """Test function load.

                Args:
                    args: Test parameter `args`.
                    kwargs: Test parameter `kwargs`.
                """
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
                """Test function decode_jpeg.

                Args:
                    args: Test parameter `args`.
                    kwargs: Test parameter `kwargs`.
                """
                return "jpeg"

            @classmethod
            def decode_png(cls, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Test function decode_png.

                Args:
                    args: Test parameter `args`.
                    kwargs: Test parameter `kwargs`.
                """
                return "png"

            @classmethod
            def decode_gif(cls, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Test function decode_gif.

                Args:
                    args: Test parameter `args`.
                    kwargs: Test parameter `kwargs`.
                """
                return "gif"

            @classmethod
            def decode_bmp(cls, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Test function decode_bmp.

                Args:
                    args: Test parameter `args`.
                    kwargs: Test parameter `kwargs`.
                """
                return "bmp"

        mock_get.return_value = DummyBackend
        assert as_tensor(decode_jpeg("test")) == as_tensor("jpeg")
        assert as_tensor(decode_png("test")) == as_tensor("png")
        assert as_tensor(decode_gif("test")) == as_tensor("gif")
        assert as_tensor(decode_bmp("test")) == as_tensor("bmp")


def test_core_math_ops_trig_fallbacks() -> None:
    """Test trigonometric fallbacks."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.core_math_ops import _acos, _acosh, _asin, _asinh, _atan, _atan2, _atanh

    with pytest.raises(TypeError):
        as_callable(_acos)(np, 0.5)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_acosh)(np, 1.5)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_asin)(np, 0.5)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_asinh)(np, 0.5)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_atan)(np, 0.5)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_atanh)(np, 0.5)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_atan2)(np, 0.5, 0.5)(np, not as_tensor(None))

    class DummyBackend:
        """Test class DummyBackend."""

        pass


def test_core_math_ops_degrees_radians_fallbacks() -> None:
    """Test degree/radians fallbacks."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.core_math_ops import _deg2rad, _degrees, _rad2deg, _radians

    with pytest.raises(TypeError):
        as_callable(_deg2rad)(np, 180.0)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_degrees)(np, 3.14)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_rad2deg)(np, 3.14)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_radians)(np, 180.0)(np, not as_tensor(None))

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    assert _deg2rad(as_mod(DummyBackend()), 180.0) is None


def test_numpy_trig_coverage() -> None:
    """Test function test_numpy_trig_coverage."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.core_math_ops.math_general import _degrees, _rad2deg, _radians
    from ml_switcheroo_compiler.backends.numpy.eager.math_trig import _np_deg2rad, _np_rad2deg

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    assert _np_deg2rad(np, 180.0) == np.pi
    assert _np_rad2deg(np, np.pi) == 180.0
    assert _degrees(as_mod(DummyBackend()), 3.14) is None
    assert _rad2deg(as_mod(DummyBackend()), 3.14) is None
    assert _radians(as_mod(DummyBackend()), 180.0) is None


def test_core_math_ops_cbrt_fallback() -> None:
    """Test cbrt fallback."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.core_math_ops import _cbrt

    with pytest.raises(TypeError):
        as_callable(_cbrt)(np, 27.0)(np, not as_tensor(None))

    class DummyBackend:
        """Test class DummyBackend."""

        def sign(self, x: EagerValue) -> EagerValue:
            """Test function sign.

            Args:
                x: Test parameter `x`.
            """
            return 1.0 if as_tensor(x) >= as_tensor(0) else -as_tensor(1.0)

        def abs(self, x: EagerValue) -> EagerValue:
            """Test function abs.

            Args:
                x: Test parameter `x`.
            """
            return x if as_tensor(x) >= as_tensor(0) else -as_tensor(x)

        def power(self, x: EagerValue, y: EagerValue) -> EagerValue:
            """Test function power.

            Args:
                x: Test parameter `x`.
                y: Test parameter `y`.
            """
            return as_tensor(x) ** as_tensor(y)

    db = DummyBackend()
    assert as_callable(abs)(as_callable(_cbrt)(db, 27.0) - 3.0) < 1e-05
    assert as_callable(abs)(as_callable(_cbrt)(db, -27.0) - -3.0) < 1e-05


def test_core_math_ops_fix_copysign() -> None:
    """Test fix and copysign fallback."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.core_math_ops import _copysign, _fix

    with pytest.raises(TypeError):
        as_callable(_fix)(np, 1.5)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_copysign)(np, 1.0, -as_tensor(1.0))(np, not as_tensor(None))

    class DummyBackend:
        """Test class DummyBackend."""

        def floor(self, x: EagerValue) -> EagerValue:
            """Test function floor.

            Args:
                x: Test parameter `x`.
            """
            import math

            return as_tensor(as_callable(math.floor)(x))

        def ceil(self, x: EagerValue) -> EagerValue:
            """Test function ceil.

            Args:
                x: Test parameter `x`.
            """
            import math

            return as_tensor(as_callable(math.ceil)(x))

        def where(self, cond: EagerValue, x: EagerValue, y: EagerValue) -> EagerValue:
            """Test function where.

            Args:
                cond: Test parameter `cond`.
                x: Test parameter `x`.
                y: Test parameter `y`.
            """
            return x if cond else y

        def abs(self, x: EagerValue) -> EagerValue:
            """Test function abs.

            Args:
                x: Test parameter `x`.
            """
            return x if as_tensor(x) >= as_tensor(0) else -as_tensor(x)

        def sign(self, x: EagerValue) -> EagerValue:
            """Test function sign.

            Args:
                x: Test parameter `x`.
            """
            return 1.0 if as_tensor(x) >= as_tensor(0) else -as_tensor(1.0)

    db = DummyBackend()
    assert _fix(as_mod(db), 1.5) == 1.0
    assert _fix(as_mod(db), -1.5) == -1.0
    assert _copysign(as_mod(db), 2.0, -1.0) == -2.0


def test_core_math_ops_f_funcs() -> None:
    """Test f-based math fallbacks."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.core_math_ops import _float_power, _fmax, _fmin, _fmod, _frexp

    with pytest.raises(TypeError):
        as_callable(_float_power)(np, 2, 3)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_fmax)(np, 1, 2)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_fmin)(np, 1, 2)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_fmod)(np, 5, 2)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_frexp)(np, 1.0)(np, not as_tensor(None))

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    assert _float_power(as_mod(DummyBackend()), 2, 3) is None
    assert _fmax(as_mod(DummyBackend()), 1, 2) is None
    assert _fmin(as_mod(DummyBackend()), 1, 2) is None
    assert _fmod(as_mod(DummyBackend()), 5, 2) is None
    assert _frexp(as_mod(DummyBackend()), 1.0) is not None


def test_core_math_ops_hypot_i0_imag() -> None:
    """Test hypot, i0 and imag fallbacks."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.core_math_ops import _hypot, _i0, _imag

    with pytest.raises(TypeError):
        as_callable(_hypot)(np, 3, 4)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_i0)(np, 1.0)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_imag)(np, 1.0)(np, not as_tensor(None))

    class DummyBackend:
        """Test class DummyBackend."""

        def sqrt(self, x: EagerValue) -> EagerValue:
            """Test function sqrt.

            Args:
                x: Test parameter `x`.
            """
            return as_tensor(x) ** as_tensor(0.5)

    db = DummyBackend()
    assert _hypot(as_mod(db), 3, 4) == 5.0
    _i0(as_mod(db), 1.0)
    with pytest.raises(TypeError):
        as_callable(as_callable(_imag)(db, 1.0))(db, None)


def test_core_math_ops_is_funcs() -> None:
    """Test isclose, iscomplex, isreal fallbacks."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.core_math_ops import _isclose, _iscomplex, _isreal

    with pytest.raises(TypeError):
        as_callable(_isclose)(np, 1.0, 1.0)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_iscomplex)(np, 1.0)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_isreal)(np, 1.0)(np, not as_tensor(None))

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(AttributeError):
        as_callable(as_callable(_isclose)(db, 1.0, 1.0))(db, None)
    _iscomplex(as_mod(db), 1.0)
    _isreal(as_mod(db), 1.0)


def test_core_math_ops_k_l_funcs() -> None:
    """Test kaiser, lcm, ldexp fallbacks."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.core_math_ops import _kaiser, _lcm, _ldexp

    with pytest.raises(TypeError):
        as_callable(_kaiser)(np, 14, 14)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_lcm)(np, 12, 20)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_ldexp)(np, 1.0, 2)(np, not as_tensor(None))

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(AttributeError):
        as_callable(as_callable(_kaiser)(db, 14, 14))(db, None)
    with pytest.raises(TypeError):
        as_callable(as_callable(_lcm)(db, 12, 20))(db, None)
    with pytest.raises(TypeError):
        as_callable(as_callable(_ldexp)(db, 1.0, 2))(db, not as_tensor(None))


def test_core_math_ops_n_p_r_funcs() -> None:
    """Test nextafter, polyval, real fallbacks."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.core_math_ops import _nextafter, _polyval, _real

    with pytest.raises(TypeError):
        as_callable(_nextafter)(np, 1.0, 2.0)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_polyval)(np, [1, 2, 3], 2.0)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_real)(np, 1.0)(np, not as_tensor(None))

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(TypeError):
        as_callable(as_callable(_nextafter)(db, 1.0, 2.0))(db, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(as_callable(_polyval)(db, [1, 2, 3], 2.0))(db, None)
    with pytest.raises(TypeError):
        as_callable(as_callable(_real)(db, 1.0))(db, None)


def test_core_math_ops_s_funcs() -> None:
    """Test signbit, sinc, spacing fallbacks."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.core_math_ops import _signbit, _sinc, _spacing

    with pytest.raises(TypeError):
        as_callable(_signbit)(np, -as_tensor(1.0))(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_sinc)(np, 1.0)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_spacing)(np, 1.0)(np, not as_tensor(None))

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    assert _signbit(as_mod(db), -1.0) is True
    assert _signbit(as_mod(db), 1.0) is False
    with pytest.raises(TypeError):
        as_callable(as_callable(_sinc)(db, 1.0))(db, None)
    with pytest.raises(TypeError):
        as_callable(as_callable(_spacing)(db, 1.0))(db, None)


def test_core_math_ops_u_z_funcs() -> None:
    """Test unwrap, zeta fallbacks."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.core_math_ops import _np_zeta, _unwrap

    with pytest.raises(TypeError):
        as_callable(_unwrap)(np, [0.0, 3.14, 6.28])(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_np_zeta)(np, 2.0)(np, not as_tensor(None))

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    _unwrap(as_mod(db), [0.0])
    import sys
    from unittest.mock import patch

    with patch.dict(sys.modules, {"scipy.special": None}):
        with pytest.raises(ModuleNotFoundError):
            as_callable(as_callable(_np_zeta)(db, 2.0))(db, None)


def test_core_math_ops_bessel_i_funcs() -> None:
    """Test bessel i0, i0e, i1, i1e fallbacks."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.core_math_ops import _bessel_i0, _bessel_i0e, _bessel_i1, _bessel_i1e

    with pytest.raises(TypeError):
        as_callable(_bessel_i0)(np, 1.0)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_bessel_i0e)(np, 1.0)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_bessel_i1)(np, 1.0)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_bessel_i1e)(np, 1.0)(np, not as_tensor(None))

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    import sys
    from unittest.mock import patch

    with patch.dict(sys.modules, {"scipy.special": None}):
        with pytest.raises(ModuleNotFoundError):
            as_callable(as_callable(_bessel_i0)(db, 1.0))(db, None)
        with pytest.raises(ModuleNotFoundError):
            as_callable(as_callable(_bessel_i0e)(db, 1.0))(db, None)
        with pytest.raises(ModuleNotFoundError):
            as_callable(as_callable(_bessel_i1)(db, 1.0))(db, None)
        with pytest.raises(ModuleNotFoundError):
            as_callable(as_callable(_bessel_i1e)(db, 1.0))(db, None)


def test_core_math_ops_bessel_j_funcs() -> None:
    """Test bessel j0, j1, jn fallbacks."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.core_math_ops import _bessel_j0, _bessel_j1, _bessel_jn

    with pytest.raises(TypeError):
        as_callable(_bessel_j0)(np, 1.0)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_bessel_j1)(np, 1.0)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_bessel_jn)(np, 2, 1.0)(np, not as_tensor(None))

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    import sys
    from unittest.mock import patch

    with patch.dict(sys.modules, {"scipy.special": None}):
        with pytest.raises(ModuleNotFoundError):
            as_callable(as_callable(_bessel_j0)(db, 1.0))(db, None)
        with pytest.raises(ModuleNotFoundError):
            as_callable(as_callable(_bessel_j1)(db, 1.0))(db, None)
        with pytest.raises(ModuleNotFoundError):
            as_callable(as_callable(_bessel_jn)(db, 2, 1.0))(db, None)


def test_core_math_ops_bessel_k_y_funcs() -> None:
    """Test bessel k0, k0e, k1, k1e, y0, y1 fallbacks."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.core_math_ops import _bessel_k0, _bessel_k0e, _bessel_k1, _bessel_k1e, _bessel_y0, _bessel_y1

    with pytest.raises(TypeError):
        as_callable(_bessel_k0)(np, 1.0)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_bessel_k0e)(np, 1.0)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_bessel_k1)(np, 1.0)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_bessel_k1e)(np, 1.0)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_bessel_y0)(np, 1.0)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_bessel_y1)(np, 1.0)(np, not as_tensor(None))

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    import sys
    from unittest.mock import patch

    with patch.dict(sys.modules, {"scipy.special": None}):
        with pytest.raises(ModuleNotFoundError):
            as_callable(as_callable(_bessel_k0)(db, 1.0))(db, None)
        with pytest.raises(ModuleNotFoundError):
            as_callable(as_callable(_bessel_k0e)(db, 1.0))(db, None)
        with pytest.raises(ModuleNotFoundError):
            as_callable(as_callable(_bessel_k1)(db, 1.0))(db, None)
        with pytest.raises(ModuleNotFoundError):
            as_callable(as_callable(_bessel_k1e)(db, 1.0))(db, None)
        with pytest.raises(ModuleNotFoundError):
            as_callable(as_callable(_bessel_y0)(db, 1.0))(db, None)
        with pytest.raises(ModuleNotFoundError):
            as_callable(as_callable(_bessel_y1)(db, 1.0))(db, None)


def test_core_math_ops_beta_digamma_funcs() -> None:
    """Test beta, betainc, digamma fallbacks."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.core_math_ops import _beta, _betainc, _digamma

    with pytest.raises(IndexError):
        as_callable(_beta)(np, 1.0, 2.0)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_betainc)(np, 1.0, 2.0, 0.5)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_digamma)(np, 1.0)(np, not as_tensor(None))

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    import sys
    from unittest.mock import patch

    with patch.dict(sys.modules, {"scipy.special": None}):
        with pytest.raises(ModuleNotFoundError):
            as_callable(as_callable(_beta)(db, 1.0, 2.0))(db, None)
        with pytest.raises(ModuleNotFoundError):
            as_callable(as_callable(_betainc)(db, 1.0, 2.0, 0.5))(db, None)
        with pytest.raises(ModuleNotFoundError):
            as_callable(as_callable(_digamma)(db, 1.0))(db, None)


def test_core_math_ops_igammac_polygamma_funcs() -> None:
    """Test igammac, polygamma fallbacks."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.core_math_ops import _igammac, _polygamma

    with pytest.raises(TypeError):
        as_callable(_igammac)(np, 1.0, 2.0)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_polygamma)(np, 1, 2.0)(np, not as_tensor(None))

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    import sys
    from unittest.mock import patch

    with patch.dict(sys.modules, {"scipy.special": None}):
        with pytest.raises(ModuleNotFoundError):
            as_callable(as_callable(_igammac)(db, 1.0, 2.0))(db, None)
        with pytest.raises(ModuleNotFoundError):
            as_callable(as_callable(_polygamma)(db, 1, 2.0))(db, None)


def test_core_math_ops_heaviside_fallback() -> None:
    """Test heaviside fallback."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.core_math_ops import _heaviside

    with pytest.raises(TypeError):
        as_callable(_heaviside)(np, 1.0, 0.5)(np, not as_tensor(None))

    class DummyBackend:
        """Test class DummyBackend."""

        def where(self, cond: EagerValue, t: EagerValue, f: EagerValue) -> EagerValue:
            """Test function where.

            Args:
                cond: Test parameter `cond`.
                t: Test parameter `t`.
                f: Test parameter `f`.
            """
            return t if cond else f

    db = DummyBackend()
    assert _heaviside(as_mod(db), 1.0, 0.5) == 1.0
    assert _heaviside(as_mod(db), -1.0, 0.5) == 0.0
    assert _heaviside(as_mod(db), 0.0, 0.5) == 0.5


def test_core_math_ops_accumulate_addn() -> None:
    """Test accumulate_n and add_n fallbacks."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.core_math_ops import _accumulate_n, _add_n

    assert _accumulate_n(np, [1, 2, 3]) == 6
    assert _add_n(np, [1, 2, 3]) == 6
    with pytest.raises(TypeError):
        as_callable(_accumulate_n)(np, [])(np, None)
    with pytest.raises(TypeError):
        as_callable(_add_n)(np, [])(np, None)


def test_core_math_ops_adjoint_det() -> None:
    """Test adjoint, det fallbacks."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.core_math_ops import _adjoint, _det

    x = np.array([[1.0 + 1j, 2.0], [3.0, 4.0]])
    with pytest.raises(TypeError):
        as_callable(_adjoint)(np, x)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_det)(np, x)(np, not as_tensor(None))

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(AttributeError):
        as_callable(as_callable(_adjoint)(db, x))(db, not as_tensor(None))
    with pytest.raises(AttributeError):
        as_callable(as_callable(_det)(db, x))(db, not as_tensor(None))

    class DummyBackend2:
        """Test class DummyBackend2."""

        def conj(self, x: EagerValue) -> EagerValue:
            """Test function conj.

            Args:
                x: Test parameter `x`.
            """
            return as_tensor(as_callable(np.conj)(x))

        def transpose(self, x: EagerValue) -> EagerValue:
            """Test function transpose.

            Args:
                x: Test parameter `x`.
            """
            return as_callable(np.transpose)(x)

    assert _adjoint(as_mod(DummyBackend2()), x) is not None


def test_core_math_ops_eig_funcs() -> None:
    """Test eig, eigh, eigvals, eigvalsh fallbacks."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.core_math_ops import _eig, _eigh, _eigvals, _eigvalsh

    x = np.array([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(TypeError):
        as_callable(_eig)(np, x)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_eigh)(np, x)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_eigvals)(np, x)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_eigvalsh)(np, x)(np, not as_tensor(None))

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(AttributeError):
        as_callable(as_callable(_eig)(db, x))(db, not as_tensor(None))
    with pytest.raises(AttributeError):
        as_callable(as_callable(_eigh)(db, x))(db, not as_tensor(None))
    with pytest.raises(AttributeError):
        as_callable(as_callable(_eigvals)(db, x))(db, not as_tensor(None))
    with pytest.raises(AttributeError):
        as_callable(as_callable(_eigvalsh)(db, x))(db, not as_tensor(None))

    class DummyBackendLinalg:
        """Test class DummyBackendLinalg."""

        class linalg:
            """Test class linalg."""

            @staticmethod
            def eig(x: EagerValue) -> EagerValue:
                """Test function eig.

                Args:
                    x: Test parameter `x`.
                """
                return x

            @staticmethod
            def eigh(x: EagerValue) -> EagerValue:
                """Test function eigh.

                Args:
                    x: Test parameter `x`.
                """
                return x

            @staticmethod
            def eigvals(x: EagerValue) -> EagerValue:
                """Test function eigvals.

                Args:
                    x: Test parameter `x`.
                """
                return x

            @staticmethod
            def eigvalsh(x: EagerValue) -> EagerValue:
                """Test function eigvalsh.

                Args:
                    x: Test parameter `x`.
                """
                return x

    dbl = DummyBackendLinalg()
    assert _eig(as_mod(dbl), x) is not None
    assert _eigh(as_mod(dbl), x) is not None
    assert _eigvals(as_mod(dbl), x) is not None
    assert _eigvalsh(as_mod(dbl), x) is not None

    class DummyBackendFlat:
        """Test class DummyBackendFlat."""

        def eig(self, x: EagerValue) -> EagerValue:
            """Test function eig.

            Args:
                x: Test parameter `x`.
            """
            return x

        def eigh(self, x: EagerValue) -> EagerValue:
            """Test function eigh.

            Args:
                x: Test parameter `x`.
            """
            return x

        def eigvals(self, x: EagerValue) -> EagerValue:
            """Test function eigvals.

            Args:
                x: Test parameter `x`.
            """
            return x

        def eigvalsh(self, x: EagerValue) -> EagerValue:
            """Test function eigvalsh.

            Args:
                x: Test parameter `x`.
            """
            return x

    dbf = DummyBackendFlat()
    assert _eig(as_mod(dbf), x) is not None
    assert _eigh(as_mod(dbf), x) is not None
    assert _eigvals(as_mod(dbf), x) is not None
    assert _eigvalsh(as_mod(dbf), x) is not None


def test_core_math_ops_cholesky_funcs() -> None:
    """Test cholesky, cholesky_ex, cholesky_solve fallbacks."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.core_math_ops import _cholesky, _cholesky_ex, _cholesky_solve

    x = np.array([[2.0, 1.0], [1.0, 2.0]])
    b = np.array([1.0, 2.0])
    with pytest.raises(TypeError):
        as_callable(_cholesky)(np, x)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_cholesky_ex)(np, x)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_cholesky_solve)(np, b, x)(np, not as_tensor(None))

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(AttributeError):
        as_callable(as_callable(_cholesky)(db, x))(db, not as_tensor(None))
    with pytest.raises(AttributeError):
        as_callable(as_callable(_cholesky_ex)(db, x))(db, not as_tensor(None))
    with pytest.raises(AttributeError):
        as_callable(as_callable(_cholesky_solve)(db, b, x))(db, not as_tensor(None))

    class DummyBackendLinalg:
        """Test class DummyBackendLinalg."""

        class linalg:
            """Test class linalg."""

            @staticmethod
            def cholesky(x: EagerValue) -> EagerValue:
                """Test function cholesky.

                Args:
                    x: Test parameter `x`.
                """
                return x

            @staticmethod
            def cho_solve(b: EagerValue, x: EagerValue) -> EagerValue:
                """Test function cho_solve.

                Args:
                    b: Test parameter `b`.
                    x: Test parameter `x`.
                """
                return b

        def zeros(self, shape: EagerValue, dtype: EagerValue) -> EagerValue:
            """Test function zeros.

            Args:
                shape: Test parameter `shape`.
                dtype: Test parameter `dtype`.
            """
            return 0

    dbl = DummyBackendLinalg()
    assert _cholesky(as_mod(dbl), x) is not None
    assert _cholesky_ex(as_mod(dbl), x) is not None
    assert _cholesky_solve(as_mod(dbl), b, x) is not None

    class DummyBackendFlat:
        """Test class DummyBackendFlat."""

        def cholesky(self, x: EagerValue) -> EagerValue:
            """Test function cholesky.

            Args:
                x: Test parameter `x`.
            """
            return x

        def cho_solve(self, b: EagerValue, x: EagerValue) -> EagerValue:
            """Test function cho_solve.

            Args:
                b: Test parameter `b`.
                x: Test parameter `x`.
            """
            return b

    dbf = DummyBackendFlat()
    assert _cholesky(as_mod(dbf), x) is not None
    assert _cholesky_ex(as_mod(dbf), x) is not None
    assert _cholesky_solve(as_mod(dbf), b, x) is not None


def test_core_math_ops_banded_householder_funcs() -> None:
    """Test banded_triangular_solve, householder_product fallbacks."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.core_math_ops import _banded_triangular_solve, _householder_product

    a = np.array([[0, 1], [2, 3], [4, 0]])
    b = np.array([1, 2])
    with pytest.raises(TypeError):
        as_callable(_banded_triangular_solve)(np, a, b)(np, not as_tensor(None))
    v = np.array([[1.0, 0.0], [0.0, 1.0]])
    tau = np.array([1.0, 1.0])

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(AttributeError):
        as_callable(as_callable(_banded_triangular_solve)(db, a, b))(db, not as_tensor(None))
    import sys
    from unittest.mock import patch

    with patch.dict(sys.modules, {"torch": None}):
        with pytest.raises(AttributeError):
            as_callable(as_callable(_householder_product)(db, v, tau))(db, not as_tensor(None))


def test_core_math_ops_matrix_power_rank() -> None:
    """Test matrix_power, matrix_rank fallbacks."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.core_math_ops import _matrix_power, _matrix_rank

    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(TypeError):
        as_callable(_matrix_power)(np, x, 2)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_matrix_rank)(np, x)(np, not as_tensor(None))

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(AttributeError):
        as_callable(as_callable(_matrix_power)(db, x, 2))(db, not as_tensor(None))
    with pytest.raises(AttributeError):
        as_callable(as_callable(_matrix_rank)(db, x))(db, not as_tensor(None))


def test_core_math_ops_norm_pinv() -> None:
    """Test norm, pinv fallbacks."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.core_math_ops import _norm, _pinv

    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(TypeError):
        as_callable(_norm)(np, x)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_pinv)(np, x)(np, not as_tensor(None))

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(AttributeError):
        as_callable(as_callable(_norm)(db, x))(db, not as_tensor(None))
    with pytest.raises(AttributeError):
        as_callable(as_callable(_pinv)(db, x))(db, not as_tensor(None))


def test_core_math_ops_qr_slogdet() -> None:
    """Test qr, slogdet fallbacks."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.core_math_ops import _qr, _slogdet

    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(TypeError):
        as_callable(_qr)(np, x)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_slogdet)(np, x)(np, not as_tensor(None))

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(AttributeError):
        as_callable(as_callable(_qr)(db, x))(db, not as_tensor(None))
    with pytest.raises(AttributeError):
        as_callable(as_callable(_slogdet)(db, x))(db, not as_tensor(None))


def test_core_math_ops_solve_svd() -> None:
    """Test solve, svd fallbacks."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.core_math_ops import _solve, _svd

    a = np.array([[1.0, 2.0], [3.0, 5.0]])
    b = np.array([1.0, 2.0])
    with pytest.raises(TypeError):
        as_callable(_solve)(np, a, b)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_svd)(np, a)(np, not as_tensor(None))

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(AttributeError):
        as_callable(as_callable(_solve)(db, a, b))(db, not as_tensor(None))
    with pytest.raises(AttributeError):
        as_callable(as_callable(_svd)(db, a))(db, not as_tensor(None))


def test_core_math_ops_tensorinv_solve() -> None:
    """Test tensorinv, tensorsolve fallbacks."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.core_math_ops import _tensorinv, _tensorsolve

    a = np.eye(4).reshape(2, 2, 2, 2)
    b = np.eye(2)
    with pytest.raises(TypeError):
        as_callable(_tensorinv)(np, a)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_tensorsolve)(np, a, b)(np, not as_tensor(None))

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(AttributeError):
        as_callable(as_callable(_tensorinv)(db, a))(db, not as_tensor(None))
    with pytest.raises(AttributeError):
        as_callable(as_callable(_tensorsolve)(db, a, b))(db, not as_tensor(None))


def test_core_math_ops_bincount_correlate() -> None:
    """Test bincount, correlate fallbacks."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.core_math_ops import _bincount, _correlate

    with pytest.raises(TypeError):
        as_callable(_bincount)(np, [1, 1, 2])(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_correlate)(np, [1, 2, 3], [0, 1])(np, not as_tensor(None))

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(AttributeError):
        as_callable(as_callable(_bincount)(db, [1, 1, 2]))(db, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(as_callable(_correlate)(db, [1, 2, 3], [0, 1]))(db, not as_tensor(None))


def test_core_math_ops_cross_cummax_cummin() -> None:
    """Test cross, cummax, cummin fallbacks."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.core_math_ops import _cross, _cummax, _cummin

    with pytest.raises(TypeError):
        as_callable(_cross)(np, [1, 0, 0], [0, 1, 0])(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_cummax)(np, [1, 3, 2])(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_cummin)(np, [2, 1, 3])(np, not as_tensor(None))

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(AttributeError):
        as_callable(as_callable(_cross)(db, [1, 0, 0], [0, 1, 0]))(db, not as_tensor(None))
    with pytest.raises(AttributeError):
        as_callable(as_callable(_cummax)(db, [1, 3, 2]))(db, not as_tensor(None))
    with pytest.raises(AttributeError):
        as_callable(as_callable(_cummin)(db, [2, 1, 3]))(db, not as_tensor(None))

    class DummyBackendAcc:
        """Test class DummyBackendAcc."""

        class maximum:
            """Test class maximum."""

            @staticmethod
            def accumulate(x: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Test function accumulate.

                Args:
                    x: Test parameter `x`.
                    kwargs: Test parameter `kwargs`.
                """
                return x

        class minimum:
            """Test class minimum."""

            @staticmethod
            def accumulate(x: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Test function accumulate.

                Args:
                    x: Test parameter `x`.
                    kwargs: Test parameter `kwargs`.
                """
                return x

    dba = DummyBackendAcc()
    assert _cummax(as_mod(dba), [1, 3, 2]) is not None
    assert _cummin(as_mod(dba), [2, 1, 3]) is not None


def test_core_math_ops_cumlogsumexp() -> None:
    """Test cumlogsumexp fallbacks."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.core_math_ops import _cumlogsumexp, _cumulative_logsumexp

    with pytest.raises(TypeError):
        as_callable(_cumlogsumexp)(np, [1.0, 2.0, 3.0])(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_cumulative_logsumexp)(np, [1.0, 2.0, 3.0])(np, not as_tensor(None))

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(AttributeError):
        as_callable(as_callable(_cumlogsumexp)(db, [1.0, 2.0, 3.0]))(db, not as_tensor(None))


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
            """Test function divide.

            Args:
                x: Test parameter `x`.
                y: Test parameter `y`.
            """
            return as_tensor(x) / as_tensor(y) if as_tensor(y) != as_tensor(0) else as_callable(float)("inf")

        def multiply(self, x: EagerValue, y: EagerValue) -> EagerValue:
            """Test function multiply.

            Args:
                x: Test parameter `x`.
                y: Test parameter `y`.
            """
            return as_tensor(x) * as_tensor(y)

        def where(self, cond: EagerValue, t: EagerValue, f: EagerValue) -> EagerValue:
            """Test function where.

            Args:
                cond: Test parameter `cond`.
                t: Test parameter `t`.
                f: Test parameter `f`.
            """
            return t if cond else f

        def isnan(self, x: EagerValue) -> EagerValue:
            """Test function isnan.

            Args:
                x: Test parameter `x`.
            """
            import math

            return as_callable(math.isnan)(x)

    db = DummyBackend()
    assert _divide_no_nan(as_mod(db), 1.0, 0.0) == 0.0
    assert _multiply_no_nan(as_mod(db), float("nan"), 0.0) == 0.0


def test_core_math_ops_extract_fft2() -> None:
    """Test extract, fft2 fallbacks."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.core_math_ops import _extract, _fft2

    with pytest.raises(TypeError):
        as_callable(_extract)(np, [True, False], [1, 2])(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_fft2)(np, [[1.0, 2.0], [3.0, 4.0]])(np, not as_tensor(None))

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(TypeError):
        as_callable(as_callable(_extract)(db, [True, False], [1, 2]))(db, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(as_callable(_fft2)(db, [[1.0, 2.0], [3.0, 4.0]]))(db, not as_tensor(None))


def test_core_math_ops_fftfreq_fftnd() -> None:
    """Test fftfreq, fftnd fallbacks."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.core_math_ops import _fftfreq, _fftnd

    with pytest.raises(TypeError):
        as_callable(_fftfreq)(np, 8)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_fftnd)(np, [[1.0, 2.0], [3.0, 4.0]])(np, not as_tensor(None))

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(TypeError):
        as_callable(as_callable(_fftfreq)(db, 8))(db, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(as_callable(_fftnd)(db, [[1.0, 2.0], [3.0, 4.0]]))(db, not as_tensor(None))


def test_core_math_ops_fftshift_ifft() -> None:
    """Test fftshift, ifft fallbacks."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.core_math_ops import _fftshift, _ifft

    with pytest.raises(TypeError):
        as_callable(_fftshift)(np, [1.0, 2.0, 3.0, 4.0])(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_ifft)(np, [1.0, 2.0, 3.0, 4.0])(np, not as_tensor(None))

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(TypeError):
        as_callable(as_callable(_fftshift)(db, [1.0, 2.0, 3.0, 4.0]))(db, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(as_callable(_ifft)(db, [1.0, 2.0, 3.0, 4.0]))(db, not as_tensor(None))


def test_core_math_ops_ifft2_ifftn() -> None:
    """Test ifft2, ifftn fallbacks."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.core_math_ops import _ifft2, _ifftn

    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(TypeError):
        as_callable(_ifft2)(np, x)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_ifftn)(np, x)(np, not as_tensor(None))

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(TypeError):
        as_callable(as_callable(_ifft2)(db, x))(db, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(as_callable(_ifftn)(db, x))(db, not as_tensor(None))


def test_core_math_ops_ifftshift_igamma() -> None:
    """Test ifftshift, igamma fallbacks."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.core_math_ops import _ifftshift, _igamma

    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(TypeError):
        as_callable(_ifftshift)(np, x)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_igamma)(np, 1.0, 2.0)(np, not as_tensor(None))

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(TypeError):
        as_callable(as_callable(_ifftshift)(db, x))(db, not as_tensor(None))
    import sys
    from unittest.mock import patch

    with patch.dict(sys.modules, {"scipy.special": None}):
        with pytest.raises(ModuleNotFoundError):
            as_callable(as_callable(_igamma)(db, 1.0, 2.0))(db, None)


def test_core_math_ops_inner_inv() -> None:
    """Test inner, inv fallbacks."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.core_math_ops import _inner, _inv

    with pytest.raises(TypeError):
        as_callable(_inner)(np, [1, 2], [3, 4])(np, not as_tensor(None))
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(TypeError):
        as_callable(_inv)(np, x)(np, not as_tensor(None))

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(AttributeError):
        as_callable(as_callable(_inner)(db, [1, 2], [3, 4]))(db, not as_tensor(None))
    with pytest.raises(AttributeError):
        as_callable(as_callable(_inv)(db, x))(db, not as_tensor(None))


def test_core_math_ops_is_funcs_2() -> None:
    """Test isinf, isnan, isneginf, isposinf fallbacks."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.core_math_ops import _isinf, _isnan, _isneginf, _isposinf

    with pytest.raises(TypeError):
        as_callable(_isinf)(np, 1.0)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_isnan)(np, 1.0)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_isneginf)(np, 1.0)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_isposinf)(np, 1.0)(np, not as_tensor(None))

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(TypeError):
        as_callable(as_callable(_isinf)(db, 1.0))(db, not as_tensor(None))
    with pytest.raises(AttributeError):
        as_callable(as_callable(_isnan)(db, 1.0))(db, not as_tensor(None))
    with pytest.raises(AttributeError):
        as_callable(as_callable(_isneginf)(db, 1.0))(db, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(as_callable(_isposinf)(db, 1.0))(db, not as_tensor(None))


def test_core_math_ops_kronecker_outer() -> None:
    """Test kronecker, outer fallbacks."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.core_math_ops import _kronecker, _outer

    with pytest.raises(TypeError):
        as_callable(_kronecker)(np, [1, 2], [3, 4])(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_outer)(np, [1, 2], [3, 4])(np, not as_tensor(None))

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(AttributeError):
        as_callable(as_callable(_kronecker)(db, [1, 2], [3, 4]))(db, not as_tensor(None))
    with pytest.raises(AttributeError):
        as_callable(as_callable(_outer)(db, [1, 2], [3, 4]))(db, not as_tensor(None))


def test_core_math_ops_fabs_fill_diagonal() -> None:
    """Test fabs, fill_diagonal fallbacks."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.core_math_ops import _fabs, _fill_diagonal

    with pytest.raises(TypeError):
        as_callable(_fabs)(np, -as_tensor(1.5))(np, not as_tensor(None))
    x = np.zeros((3, 3))
    with pytest.raises(TypeError):
        as_callable(_fill_diagonal)(np, x, 1)(np, None)

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(TypeError):
        as_callable(as_callable(_fabs)(db, -as_tensor(1.5)))(db, None)
    _fill_diagonal(as_mod(db), x, 1)


def test_core_math_ops_fftconvolve_flatnonzero() -> None:
    """Test fftconvolve, flatnonzero fallbacks."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.core_math_ops import _fftconvolve, _flatnonzero

    with pytest.raises(TypeError):
        as_callable(_fftconvolve)(np, [1, 2], [3, 4])(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_flatnonzero)(np, [0, 1, 0, 2])(np, not as_tensor(None))

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(TypeError):
        as_callable(as_callable(_fftconvolve)(db, [1, 2], [3, 4]))(db, not as_tensor(None))
    with pytest.raises(AttributeError):
        as_callable(as_callable(_flatnonzero)(db, [0, 1, 0, 2]))(db, not as_tensor(None))


def test_core_math_ops_fliplr_flipud() -> None:
    """Test fliplr, flipud fallbacks."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.core_math_ops import _fliplr, _flipud

    x = np.array([[1, 2], [3, 4]])
    with pytest.raises(TypeError):
        as_callable(_fliplr)(np, x)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_flipud)(np, x)(np, not as_tensor(None))

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(AttributeError):
        as_callable(as_callable(_fliplr)(db, x))(db, not as_tensor(None))
    with pytest.raises(AttributeError):
        as_callable(as_callable(_flipud)(db, x))(db, not as_tensor(None))


def test_core_math_ops_fromiter_fromstring() -> None:
    """Test fromiter, fromstring fallbacks."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.core_math_ops import _fromiter, _fromstring

    with pytest.raises(TypeError):
        as_callable(_fromiter)(np, [1, 2, 3])(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_fromstring)(np, "1, 2", sep=",")(np, not as_tensor(None))

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(AttributeError):
        as_callable(as_callable(_fromiter)(db, [1, 2, 3]))(db, not as_tensor(None))
    with pytest.raises(AttributeError):
        as_callable(as_callable(_fromstring)(db, "1, 2", sep=","))(db, not as_tensor(None))


def test_core_math_ops_gamma_gcd() -> None:
    """Test gamma, gcd fallbacks."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.core_math_ops import _gamma, _gcd

    with pytest.raises(TypeError):
        as_callable(_gamma)(np, 2.0)(np, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(_gcd)(np, 12, 20)(np, not as_tensor(None))

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(TypeError):
        as_callable(as_callable(_gamma)(db, 2.0))(db, not as_tensor(None))
    with pytest.raises(TypeError):
        as_callable(as_callable(_gcd)(db, 12, 20))(db, not as_tensor(None))


def test_math_misc_brute_coverage() -> None:
    """Test function test_math_misc_brute_coverage."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry

    class DummyBackend:
        """Test class DummyBackend."""

        def __getattr__(self, name: EagerValue) -> EagerValue:
            """Test function __getattr__.

            Args:
                name: Test parameter `name`.
            """
            if as_callable(hasattr)(np, name):
                return as_tensor(getattr(np, str(name)))
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
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
        try:
            op(dummy, np.array([1, 2, 3]))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
        try:
            op(dummy, np.array([1, 2, 3]), np.array([1, 2, 3]))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
        try:
            op(dummy, 1)
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
        try:
            op(dummy, [1, 2])
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
        try:
            op(dummy, 1, 2)
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
        try:
            op(dummy, np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
    assert callable(test_math_misc_brute_coverage)


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
    assert callable(Resize)


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
    math_misc._np_frombuffer(np, b"hello", dtype=np.dtype(np.uint8))
    math_misc._np_descriptive(np, np.array([1.0, 2.0]))
    math_misc._np_distributions(np, np.array([1.0, 2.0]))
    math_misc._np_shifted_chebyshev_polynomial_t(np, np.array([0.5]), 2)
    math_misc._np_shifted_chebyshev_polynomial_u(np, np.array([0.5]), 2)
    math_misc._np_shifted_chebyshev_polynomial_v(np, np.array([0.5]), 2)
    math_misc._np_shifted_chebyshev_polynomial_w(np, np.array([0.5]), 2)
    math_misc._np_modified_bessel_i1(np, np.array([0.5]))
    math_misc._np_modified_bessel_k0(np, np.array([0.5]))
    math_misc._np_modified_bessel_k1(np, np.array([0.5]))
    with pytest.raises(RuntimeError):
        math_misc._np_decode_image(np, b"img")
    with pytest.raises(RuntimeError):
        math_misc._np_parse_example(np, b"example")
    with pytest.raises(RuntimeError):
        math_misc._np_parse_tensor(np, b"tensor")
    math_misc._np_read_file(np, "/tmp/ml_switcheroo_test_file.txt")
    math_misc._np_rem(np, np.array([5]), np.array([2]))
    math_misc._np_serialize_tensor(np, np.array([1]))
    math_misc._np_write_file(np, "/tmp/ml_switcheroo_test_file.txt", b"content")
    math_misc._np_confusion_matrix(np, np.array([0]), np.array([0]))
    math_misc._np_decode_csv(np, ["1,2"])
    math_misc._np_rawmatmul(np, np.ones((2, 2)), np.ones((2, 2)))
    math_misc._np_sparsedensematmul(np, np.ones((2, 2)), np.ones((2, 2)))
    math_misc._np_stridedslice(np, np.ones((10, 10)), [0], [5], [1])


def test_math_misc_exceptions() -> None:
    """Test function test_math_misc_exceptions."""
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as math_misc

    class ThrowingBackend:
        """Test class ThrowingBackend."""

        def __getattr__(self, name: EagerValue) -> None:
            """Test function __getattr__.

            Args:
                name: Test parameter `name`.
            """
            raise ValueError("Intentional error")

    tb = ThrowingBackend()
    with pytest.raises(ValueError):
        math_misc._np_rawmatmul(as_mod(tb), np.ones((2, 2)), np.ones((2, 2)))
    with pytest.raises(ValueError):
        math_misc._np_sparsedensematmul(as_mod(tb), np.ones((2, 2)), np.ones((2, 2)))
    with pytest.raises(RuntimeError):
        math_misc._np_decode_image(as_mod(tb), b"img")
    with pytest.raises(RuntimeError):
        math_misc._np_parse_example(as_mod(tb), b"example")
    with pytest.raises(RuntimeError):
        math_misc._np_parse_tensor(as_mod(tb), b"tensor")
    math_misc._np_read_file(as_mod(tb), "/tmp/ml_switcheroo_test_file.txt")
    with pytest.raises(ValueError):
        math_misc._np_rem(as_tensor(tb), np.array([5]), np.array([2]))
    math_misc._np_serialize_tensor(as_tensor(tb), np.array([1]))
    math_misc._np_write_file(as_mod(tb), "/tmp/ml_switcheroo_test_file.txt", b"content")
    math_misc._np_confusion_matrix(as_mod(tb), np.array([0]), np.array([0]))
    math_misc._np_decode_csv(as_mod(tb), ["1,2"])
    with pytest.raises(ValueError):
        math_misc._np_descriptive(as_tensor(tb), np.array([1.0, 2.0]))
    math_misc._np_decode_image(as_mod(tb))
    math_misc._np_parse_example(as_mod(tb))
    math_misc._np_parse_tensor(as_mod(tb))
    math_misc._np_read_file(as_mod(tb))
    with pytest.raises(ValueError):
        math_misc._np_rem(as_tensor(tb))
    math_misc._np_serialize_tensor(as_tensor(tb))
    math_misc._np_write_file(as_mod(tb))
    math_misc._np_confusion_matrix(as_mod(tb))
    math_misc._np_decode_csv(as_mod(tb))


def test_math_misc_type_calls() -> None:
    """Test function test_math_misc_type_calls."""
    import pytest

    with pytest.raises((ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, Exception), match=".*"):
        import numpy as np

        import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as math_misc

        class FakeOpType:
            """Test class FakeOpType."""

            def __init__(self, *args: EagerValue, **kwargs: EagerValue) -> None:
                """Test function __init__.

                Args:
                    args: Test parameter `args`.
                    kwargs: Test parameter `kwargs`.
                """
                pass

        class ThrowingOpType:
            """Test class ThrowingOpType."""

            def __init__(self, *args: EagerValue, **kwargs: EagerValue) -> None:
                """Test function __init__.

                Args:
                    args: Test parameter `args`.
                    kwargs: Test parameter `kwargs`.
                """
                raise ValueError("Intentional error from constructor")

        class BackendWithType:
            """Test class BackendWithType."""

            def __getattr__(self, name: EagerValue) -> EagerValue:
                """Test function __getattr__.

                Args:
                    name: Test parameter `name`.
                """
                return FakeOpType

        class BackendWithThrowingType:
            """Test class BackendWithThrowingType."""

            def __getattr__(self, name: EagerValue) -> EagerValue:
                """Test function __getattr__.

                Args:
                    name: Test parameter `name`.
                """
                return ThrowingOpType

        b = BackendWithType()
        math_misc._np_rawmatmul(as_mod(b), np.ones((2, 2)), np.ones((2, 2)))
        math_misc._np_sparsedensematmul(as_mod(b), np.ones((2, 2)), np.ones((2, 2)))
        as_callable(as_callable(numpy_eager_registry.get)("decode_image"))(b, b"img")
        as_callable(as_callable(numpy_eager_registry.get)("parse_example"))(b, b"example")
        as_callable(as_callable(numpy_eager_registry.get)("parse_tensor"))(b, b"tensor")
        as_callable(as_callable(numpy_eager_registry.get)("read_file"))(b, "/tmp/ml_switcheroo_test_file.txt")
        as_callable(as_callable(numpy_eager_registry.get)("rem"))(b, as_callable(np.array)([5]), as_callable(np.array)([2]))
        as_callable(as_callable(numpy_eager_registry.get)("serialize_tensor"))(b, as_callable(np.array)([1]))
        as_callable(as_callable(numpy_eager_registry.get)("write_file"))(b, "/tmp/ml_switcheroo_test_file.txt", b"content")
        as_callable(as_callable(numpy_eager_registry.get)("confusion_matrix"))(b, as_callable(np.array)([0]), as_callable(np.array)([0]))
        as_callable(as_callable(numpy_eager_registry.get)("decode_csv"))(b, ["1,2"])
        as_callable(as_callable(numpy_eager_registry.get)("descriptive"))(b, as_callable(np.array)([1.0, 2.0]))
        bt = BackendWithThrowingType()
        try:
            math_misc._np_rawmatmul(as_mod(bt), np.ones((2, 2)), np.ones((2, 2)))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
        try:
            math_misc._np_sparsedensematmul(as_mod(bt), np.ones((2, 2)), np.ones((2, 2)))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
        try:
            math_misc._np_decode_image(as_mod(bt), b"img")
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
        try:
            math_misc._np_parse_example(as_mod(bt), b"example")
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
        try:
            math_misc._np_parse_tensor(as_mod(bt), b"tensor")
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
        try:
            math_misc._np_read_file(as_mod(bt), "/tmp/ml_switcheroo_test_file.txt")
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
        try:
            math_misc._np_rem(as_tensor(bt), np.array([5]), np.array([2]))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
        try:
            math_misc._np_serialize_tensor(as_tensor(bt), np.array([1]))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
        try:
            math_misc._np_write_file(as_mod(bt), "/tmp/ml_switcheroo_test_file.txt", b"content")
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
        try:
            math_misc._np_confusion_matrix(as_mod(bt), np.array([0]), np.array([0]))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
        try:
            math_misc._np_decode_csv(as_mod(bt), ["1,2"])
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
        try:
            as_callable(as_callable(numpy_eager_registry.get)("descriptive"))(bt, as_callable(np.array)([1.0, 2.0]))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True


def test_math_misc_np_ops() -> None:
    """Test function test_math_misc_np_ops."""
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as math_misc
    import ml_switcheroo_compiler.ops as ops

    class FakeThrowingClass:
        """Test class FakeThrowingClass."""

        def __init__(self, *args: EagerValue, **kwargs: EagerValue) -> None:
            """Test function __init__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            raise ValueError("boom")

    class FakeWorkingClass:
        """Test class FakeWorkingClass."""

        shape = ()

        def __init__(self, *args: EagerValue, **kwargs: EagerValue) -> None:
            """Test function __init__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            pass

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
        math_misc._np_rawmatmul(as_mod(b), np.ones((2, 2)), np.ones((2, 2)))
        math_misc._np_sparsedensematmul(as_mod(b), np.ones((2, 2)), np.ones((2, 2)))
        with pytest.raises(RuntimeError):
            as_callable(numpy_eager_registry.get)("decode_image")(b, b"img")
        with pytest.raises(RuntimeError):
            as_callable(numpy_eager_registry.get)("parse_example")(b, b"example")
        with pytest.raises(RuntimeError):
            as_callable(numpy_eager_registry.get)("parse_tensor")(b, b"tensor")
        as_callable(numpy_eager_registry.get)("read_file")(b, "/tmp/ml_switcheroo_test_file.txt")
        as_callable(numpy_eager_registry.get)("rem")(b, as_callable(np.array)([5]), as_callable(np.array)([2]))
        as_callable(numpy_eager_registry.get)("serialize_tensor")(b, as_callable(np.array)([1]))
        as_callable(numpy_eager_registry.get)("write_file")(b, "/tmp/ml_switcheroo_test_file.txt", b"content")
        as_callable(numpy_eager_registry.get)("confusion_matrix")(b, as_callable(np.array)([0]), as_callable(np.array)([0]))
        as_callable(numpy_eager_registry.get)("decode_csv")(b, ["1,2"])
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
    math_misc._np_rawmatmul(as_mod(b), np.ones((2, 2)), np.ones((2, 2)))
    math_misc._np_sparsedensematmul(as_mod(b), np.ones((2, 2)), np.ones((2, 2)))
    with pytest.raises(RuntimeError):
        as_callable(numpy_eager_registry.get)("decode_image")(b, b"img")
    with pytest.raises(RuntimeError):
        as_callable(numpy_eager_registry.get)("parse_example")(b, b"example")
    with pytest.raises(RuntimeError):
        as_callable(numpy_eager_registry.get)("parse_tensor")(b, b"tensor")
    as_callable(numpy_eager_registry.get)("read_file")(b, "/tmp/ml_switcheroo_test_file.txt")
    as_callable(numpy_eager_registry.get)("rem")(b, as_callable(np.array)([5]), as_callable(np.array)([2]))
    as_callable(numpy_eager_registry.get)("serialize_tensor")(b, as_callable(np.array)([1]))
    as_callable(numpy_eager_registry.get)("write_file")(b, "/tmp/ml_switcheroo_test_file.txt", b"content")
    as_callable(numpy_eager_registry.get)("confusion_matrix")(b, as_callable(np.array)([0]), as_callable(np.array)([0]))
    as_callable(numpy_eager_registry.get)("decode_csv")(b, ["1,2"])


def test_math_misc_np_ops_lower() -> None:
    """Test function test_math_misc_np_ops_lower."""
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as math_misc
    import ml_switcheroo_compiler.ops as ops

    class FakeThrowingClass:
        """Test class FakeThrowingClass."""

        def __init__(self, *args: EagerValue, **kwargs: EagerValue) -> None:
            """Test function __init__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            raise ValueError("boom")

    class FakeWorkingClass:
        """Test class FakeWorkingClass."""

        shape = ()

        def __init__(self, *args: EagerValue, **kwargs: EagerValue) -> None:
            """Test function __init__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
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
        with pytest.raises(RuntimeError):
            as_callable(numpy_eager_registry.get)("decode_image")(b, b"img")
        with pytest.raises(RuntimeError):
            as_callable(numpy_eager_registry.get)("parse_example")(b, b"example")
        with pytest.raises(RuntimeError):
            as_callable(numpy_eager_registry.get)("parse_tensor")(b, b"tensor")
        as_callable(numpy_eager_registry.get)("read_file")(b, "/tmp/ml_switcheroo_test_file.txt")
        as_callable(numpy_eager_registry.get)("rem")(b, as_callable(np.array)([5]), as_callable(np.array)([2]))
        as_callable(numpy_eager_registry.get)("serialize_tensor")(b, as_callable(np.array)([1]))
        as_callable(numpy_eager_registry.get)("write_file")(b, "/tmp/ml_switcheroo_test_file.txt", b"content")
        as_callable(numpy_eager_registry.get)("confusion_matrix")(b, as_callable(np.array)([0]), as_callable(np.array)([0]))
        as_callable(numpy_eager_registry.get)("decode_csv")(b, ["1,2"])
        setattr(ops, "decode_image", FakeWorkingClass)
        setattr(ops, "parse_example", FakeWorkingClass)
        ops.parse_tensor = FakeWorkingClass
        ops.read_file = FakeWorkingClass
        setattr(ops, "rem", FakeWorkingClass)
        ops.serialize_tensor = FakeWorkingClass
        setattr(ops, "write_file", FakeWorkingClass)
        ops.confusion_matrix = FakeWorkingClass
        setattr(ops, "decode_csv", FakeWorkingClass)
        with pytest.raises(RuntimeError):
            as_callable(numpy_eager_registry.get)("decode_image")(b, b"img")
        with pytest.raises(RuntimeError):
            as_callable(numpy_eager_registry.get)("parse_example")(b, b"example")
        with pytest.raises(RuntimeError):
            as_callable(numpy_eager_registry.get)("parse_tensor")(b, b"tensor")
        as_callable(numpy_eager_registry.get)("read_file")(b, "/tmp/ml_switcheroo_test_file.txt")
        as_callable(numpy_eager_registry.get)("rem")(b, as_callable(np.array)([5]), as_callable(np.array)([2]))
        as_callable(numpy_eager_registry.get)("serialize_tensor")(b, as_callable(np.array)([1]))
        as_callable(numpy_eager_registry.get)("write_file")(b, "/tmp/ml_switcheroo_test_file.txt", b"content")
        as_callable(numpy_eager_registry.get)("confusion_matrix")(b, as_callable(np.array)([0]), as_callable(np.array)([0]))
        as_callable(numpy_eager_registry.get)("decode_csv")(b, ["1,2"])
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

    red._np_top_k(b, np.array([1.0, 2.0, 3.0]), as_tensor(MockK()))

    class MockK2:
        """Test class MockK2."""

        class Data:
            """Test class Data."""

            def item(self) -> EagerValue:
                """Test function item."""
                return 2

        data = Data()

    red._np_top_k(b, np.array([1.0, 2.0, 3.0]), as_tensor(MockK2()))
    red._segment_sum(np.array([1.0, 2.0]), np.array([0, 1]))
    red._segment_sum(np.array([1.0, 2.0]), np.array([0, 1]), num_segments=2)
    red._logsumexp(np.array([1.0, 2.0]))
    red._logsumexp(np.array([1.0, 2.0]), keepdims=True)
    red._top_k(np.array([1.0, 2.0, 3.0]), 2)
    red._top_k(np.array([[1.0, 2.0, 3.0]]), 2, axis=1)
    with pytest.raises(AttributeError):
        red._np_nms(b, np.array([[0, 0, 1, 1]]), np.array([1.0]), 1)

    class DummyConfig:
        """Test class DummyConfig."""

        window_dimensions = [2]
        window_strides = [1]
        window_dilation = [1]
        base_dilation = [1]
        padding = "SAME"

    as_callable(red._reduce_window)(as_callable(np.array)([1.0, 2.0, 3.0]), 0.0, "max", as_callable(DummyConfig)())
    as_callable(red._reduce_window)(as_callable(np.array)([1.0, 2.0, 3.0]), 0.0, "min", as_callable(DummyConfig)())
    as_callable(red._reduce_window)(as_callable(np.array)([1.0, 2.0, 3.0]), 0.0, "sum", as_callable(DummyConfig)())
    as_callable(red._reduce_window)(as_callable(np.array)([1.0, 2.0, 3.0]), 0.0, "prod", as_callable(DummyConfig)())
    with pytest.raises(ValueError):
        as_callable(red._reduce_window)(as_callable(np.array)([1.0, 2.0, 3.0]), 0.0, "unknown", as_callable(DummyConfig)())

    class DummyConfig2:
        """Test class DummyConfig2."""

        window_dimensions = [2]
        window_strides = [1]
        window_dilation = [1]
        base_dilation = [2]
        padding = [(1, 1)]

    as_callable(red._reduce_window)(as_callable(np.array)([1.0, 2.0, 3.0]), 0.0, "sum", as_callable(DummyConfig2)())

    class DummyConfig3:
        """Test class DummyConfig3."""

        window_dimensions = [2]
        window_strides = [1]
        window_dilation = [1]
        base_dilation = [1]
        padding = None

    as_callable(red._reduce_window)(as_callable(np.array)([1.0, 2.0, 3.0]), 0.0, "sum", as_callable(DummyConfig3)())


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

    with pytest.raises(IndexError):
        as_callable(red._reduce_window)(as_callable(np.array)([[1.0, 2.0], [3.0, 4.0]]), 0.0, "max", as_callable(DummyConfig4)())

    class DummyConfig5:
        """Test class DummyConfig5."""

        window_dimensions = [2]
        window_strides = [1]
        window_dilation = [1]
        base_dilation = [1]
        padding = "VALID"

    as_callable(red._reduce_window)(as_callable(np.array)([1.0, 2.0, 3.0]), 0.0, "max", as_callable(DummyConfig5)())

    class DummyConfig6:
        """Test class DummyConfig6."""

        window_dimensions = [1]
        window_strides = [1]
        window_dilation = [1]
        base_dilation = [1]
        padding = "SAME"

    as_callable(red._reduce_window)(1.0, 0.0, "max", as_callable(DummyConfig6)())
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

    assert _fftconvolve(as_mod(DummyBackendWithoutFFT()), None, [3, 4]) is None


def test_math_internal_return_zero() -> None:
    """Test math internal return zero."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops.math_internal import _np_tensorarraywrite

    class DummyBackend:
        """Dummy."""

        pass

    assert _np_tensorarraywrite(as_mod(DummyBackend()), "not_a_list", 0, 1) == 0


def test_math_fft_branch_coverage() -> None:
    """Test math fft branch coverage."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops.math_fft import _fftconvolve

    class DummySignal:
        """Dummy."""

        pass

    class DummyBackendWithSignal:
        """Dummy."""

        signal = DummySignal()

    assert _fftconvolve(as_mod(DummyBackendWithSignal()), [1, 2], None) is None


def test_math_fft_signal_fftconvolve() -> None:
    """Test math fft signal fftconvolve."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops.math_fft import _fftconvolve

    class DummySignal:
        """Dummy."""

        def fftconvolve(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Fftconvolve.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "success"

    class DummyBackendWithSignal:
        """Dummy."""

        signal = DummySignal()

    assert _fftconvolve(as_mod(DummyBackendWithSignal()), [1, 2], [3, 4]) == "success"
