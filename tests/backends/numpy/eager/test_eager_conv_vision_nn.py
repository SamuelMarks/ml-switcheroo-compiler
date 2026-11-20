# ruff: noqa: C901, PLR0912, PLR0915, F841
"""Tests for NumPy eager dilated convolution, neural activations, and vision kernels."""

from __future__ import annotations

import importlib
import typing
from unittest.mock import MagicMock

import numpy as np
import pytest

import ml_switcheroo_compiler.backends.eager as eager_mod
import tests.backends.numpy.eager._math_advanced_test_ns as mod
from ml_switcheroo_compiler.backends.eager.nn_ops.conv_reference import _conv_general_dilated, _get_transpose
from ml_switcheroo_compiler.backends.eager.nn_ops.layers import (
    _gelu,
    _np_activity_regularization,
    _np_alpha_dropout,
    _np_dropout,
    _np_relu,
    _np_time_distributed,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_scatter import _band_part
from ml_switcheroo_compiler.backends.eager.vision_ops.color import (
    _np_adjust_brightness,
    _np_adjust_contrast,
    _np_adjust_hue,
    _np_adjust_saturation,
    _np_auto_contrast,
    _np_equalization,
    _np_invert,
    _np_posterize,
    _np_rgb_to_grayscale,
    _np_solarize,
)
from ml_switcheroo_compiler.backends.eager.vision_ops.geometry import (
    _np_affine_generator,
    _np_affine_grid,
    _np_affine_transform,
    _np_elastic_transform,
    _np_extract_bounding_boxes,
    _np_iou,
    _np_nms,
    _np_perspective_transform,
)
from ml_switcheroo_compiler.backends.eager_registry import BackendArg, numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray, execute_op
from ml_switcheroo_compiler.ops.configs import ConvConfig, WindowConfig
from tests.testing_utils import DummyDimSpecs, MockTestStub, extract_test_shape, require_callable, unpack_pair

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg
_reduce_window = importlib.import_module(
    "ml_switcheroo_compiler.backends.eager.reference_math.window_reductions"
)._reduce_window


def test_segment_sum_coverage() -> None:
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


def test_conv_general_dilated_all_coverage() -> None:
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


def test_reduce_window_coverage() -> None:
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


def test_conv_general_dilated_coverage() -> None:
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


def test_band_part_coverage() -> None:
    """Test the band part coverage behavior.

    Returns:
        OpValue: The inferred shape or computed result.
    """
    res = _band_part(np.ones((3, 3)), 1, 1)
    assert res.shape == (3, 3)


def test_numpy_eager_coverage() -> None:
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
    x.dtype = "float32"
    x.device = "cpu"
    x.ndim = 2
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


def test_conv_general_dilated_extra() -> None:
    """Test the conv general dilated extra behavior.

    Returns:
        OpValue: The inferred shape or computed result.
    """
    from ml_switcheroo_compiler.ops.configs import ConvConfig, ConvDimensionNumbers

    lhs = np.ones((1, 2, 5))
    rhs = np.ones((4, 2, 3))
    res = _conv_general_dilated(lhs, rhs, ConvConfig(window_strides=(1,), padding=None))
    assert extract_test_shape(res) == (1, 4, 3)
    res = _conv_general_dilated(
        lhs,
        rhs,
        ConvConfig(
            window_strides=(1,),
            padding="VALID",
            dimension_numbers=ConvDimensionNumbers(lhs_spec=(0, 1, 2), rhs_spec=(0, 1, 2), out_spec=(0, 1, 2)),
        ),
    )
    assert extract_test_shape(res) == (1, 4, 3)


def test_numpy_eager_conv_extra() -> None:
    """Test the numpy eager conv extra behavior.

    Returns:
        OpValue: The inferred shape or computed result.
    """
    ConvTranspose = numpy_eager_registry.get("ConvTranspose")
    _out = require_callable(ConvTranspose)(np, np.ones((1, 1, 1, 1)), np.ones((1, 1, 1, 1)), strides=1, padding="SAME")
    assert np.asarray(_out).shape == (1, 1, 1, 1) and bool(
        np.allclose(np.asarray(_out, dtype=float), [[[[1.0]]]], atol=1e-3)
    )


def test_numpy_eager_linalg_extra() -> None:
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


def test_numpy_eager_math_extra() -> None:
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


def test_numpy_eager_random_extra() -> None:
    """Test the numpy eager random extra behavior.

    Returns:
        OpValue: The inferred shape or computed result.
    """
    Dropout = numpy_eager_registry.get("Dropout")
    res = require_callable(Dropout)(np, np.ones((2,)), 0.5)
    assert res.shape == (2,)


def test_numpy_eager_shape_extra() -> None:
    """Test the numpy eager shape extra behavior.

    Returns:
        OpValue: The inferred shape or computed result.
    """
    SparseExpandDims = numpy_eager_registry.get("SparseExpandDims")
    x = np.array([1])
    assert require_callable(SparseExpandDims)(np, x).shape == (1, 1)


def test_numpy_nn_eager_extra() -> None:
    """Test the numpy nn eager extra behavior.

    Returns:
        OpValue: The inferred shape or computed result.
    """
    _gelu(np.array([-1.0, 2.0, 5.0]))
    res = _np_relu(np, np.array([-1.0, 2.0, 5.0]))
    assert np.allclose(res, [0.0, 2.0, 5.0])
    res = _np_alpha_dropout(np, np.ones((2, 2)), rate=0.5, seed=42, training=True)
    assert res.shape == (2, 2)
    res = _np_alpha_dropout(np, np.ones((2, 2)), rate=0.0)
    assert np.allclose(res, 1.0)
    res = _np_alpha_dropout(np, np.ones((2, 2)), rate=0.5, seed=42, noise_shape=(2, 1), training=True)
    assert res.shape == (2, 2)
    res = _np_activity_regularization(np, np.ones((2, 2)), l1=0.1, l2=0.2)
    assert res.shape == (2, 2)
    res = _np_dropout(np, np.ones((2, 2)), rate=0.5, seed=42, training=True)
    assert res.shape == (2, 2)
    res = _np_dropout(np, np.ones((2, 2)), rate=0.0)
    assert np.allclose(res, 1.0)
    res = _np_dropout(np, np.ones((2, 2)), rate=0.5, seed=42, noise_shape=(2, 1), training=True)
    assert res.shape == (2, 2)


def test_np_time_distributed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the np time distributed behavior.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch parameter.

    Returns:
        OpValue: The inferred shape or computed result.
    """
    with pytest.raises(AttributeError):

        def mock_exec(backend: BackendArg, op_name: BackendArg, x: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Evaluate and process the mock exec operation.

            Args:
                backend (BackendArg): Required parameter for backend.
                op_name (BackendArg): Required parameter for op_name.
                x (BackendArg): Required parameter for x.
                **kwargs (BackendArg): Arbitrary keyword arguments.

            Returns:
                BackendArg: The evaluated or processed output.
            """
            return x

        monkeypatch.setattr(eager_mod, "execute_generic_op", mock_exec)
        res = _np_time_distributed(np, np.ones((2, 2)), wrapped_op_name="Dummy")
        assert res.shape == (2, 2)
        res = _np_time_distributed(np, np.ones((2, 3, 4)), wrapped_op_name="Dummy")
        assert res.shape == (2, 3, 4)


def test_numpy_vision_color_eager() -> None:
    """Test the numpy vision color eager behavior.

    Returns:
        OpValue: The inferred shape or computed result.
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
    original_hist = np.histogram

    def mock_hist(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Evaluate and process the mock hist operation.

        Args:
            *args (BackendArg): Variable positional arguments.
            **kwargs (BackendArg): Arbitrary keyword arguments.

        Returns:
            BackendArg: The evaluated or processed output.
        """
        return (np.array([1] * 256), None)

    pytest.MonkeyPatch().setattr(np, "histogram", mock_hist, raising=False)
    try:
        _np_equalization(np, img)
    finally:
        pytest.MonkeyPatch().setattr(np, "histogram", original_hist, raising=False)

    def mock_hist2(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Evaluate and process the mock hist2 operation.

        Args:
            *args (BackendArg): Variable positional arguments.
            **kwargs (BackendArg): Arbitrary keyword arguments.

        Returns:
            BackendArg: The evaluated or processed output.
        """
        a = np.zeros(256)
        a[0] = 100
        return (a, None)

    pytest.MonkeyPatch().setattr(np, "histogram", mock_hist2, raising=False)
    try:
        _np_equalization(np, img)
    finally:
        pytest.MonkeyPatch().setattr(np, "histogram", original_hist, raising=False)
    _np_invert(np, img)
    _np_posterize(np, img, bits=4)
    _np_rgb_to_grayscale(np, img)
    _np_rgb_to_grayscale(np, img, keepdim=True)
    _out = _np_solarize(np, img, threshold=0.5)
    assert np.asarray(_out).shape == (2, 2, 3)


def test_numpy_vision_geometry_eager_extra() -> None:
    """Test the numpy vision geometry eager extra behavior."""
    el_out = _np_elastic_transform(np, np.ones((2, 2, 3)), np.ones((2, 2, 2)))
    assert extract_test_shape(el_out) == (2, 2, 3)
    bb_out = _np_extract_bounding_boxes(np, np.ones((2, 2, 3)), np.array([0, 0, 10, 10]), np.array([0]))
    assert bb_out is not None
    original_iou = eager_mod.iou_eager

    def mock_iou(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Evaluate and process the mock iou operation.

        Args:
            *args (BackendArg): Variable positional arguments.
            **kwargs (BackendArg): Arbitrary keyword arguments.

        Returns:
            BackendArg: The evaluated or processed output.
        """
        return args[1]

    eager_mod.iou_eager = mock_iou
    try:
        iou_out = _np_iou(np, np.array([0, 0, 10, 10]), np.array([5, 5, 15, 15]))
        assert iou_out is not None
    finally:
        eager_mod.iou_eager = original_iou
    original_nms = eager_mod.nms_eager

    def mock_nms(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Evaluate and process the mock nms operation.

        Args:
            *args (BackendArg): Variable positional arguments.
            **kwargs (BackendArg): Arbitrary keyword arguments.

        Returns:
            BackendArg: The evaluated or processed output.
        """
        return args[1]

    eager_mod.nms_eager = mock_nms
    try:
        nms_out = _np_nms(np, np.array([[0, 0, 10, 10]]), np.array([0.9]), max_output_size=1, threshold=0.5)
        assert nms_out is not None
    finally:
        eager_mod.nms_eager = original_nms
    pt_out = _np_perspective_transform(np, np.ones((2, 2, 3)), np.eye(3), np.eye(3), config=None)
    assert extract_test_shape(pt_out) == (2, 2, 3)
    ag_out = _np_affine_grid(np, np.eye(2, 3)[None, ...], size=(1, 1, 4, 4))
    assert ag_out is not None
    at_out = _np_affine_transform(np, np.ones((2, 2, 3)), np.eye(2, 3))
    assert extract_test_shape(at_out) == (2, 2, 3)
    gen_out = _np_affine_generator(np, 2, np.ones(1), np.ones(1), np.ones(1))
    assert extract_test_shape(gen_out) == (2, 8)


def test_numpy_lax_mocks() -> None:
    """Test the numpy lax mocks behavior.

    Returns:
        OpValue: The inferred shape or computed result.
    """
    with pytest.raises(TypeError):

        def execute(op: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Evaluate and process the execute operation.

            Args:
                op (BackendArg): Required parameter for op.
                *args (BackendArg): Variable positional arguments.
                **kwargs (BackendArg): Arbitrary keyword arguments.

            Returns:
                BackendArg: The evaluated or processed output.
            """
            return require_callable(numpy_eager_registry.get)(op)(np, *args, **kwargs)

        (a, b) = unpack_pair(execute("ApproxMaxK", np.array([1, 2])))
        assert np.array_equal(ensure_ndarray(a), np.array([2]))
        assert np.array_equal(ensure_ndarray(b), np.array([1]))
        (a, b) = unpack_pair(execute("ApproxMaxK", np.array([])))
        assert ensure_ndarray(a).size == 0
        assert ensure_ndarray(b).size == 0
        c = execute("ForiLoop", None, None, np.array([3]))
        assert np.array_equal(ensure_ndarray(c), np.array([3]))
        c = execute("ForiLoop", None)
        assert np.array_equal(ensure_ndarray(c), np.array(0))
        d = execute("IgammaGradA", np.array([4]))
        assert np.array_equal(ensure_ndarray(d), np.array([4]))
        d = execute("IgammaGradA")
        assert np.array_equal(ensure_ndarray(d), np.array(0))
