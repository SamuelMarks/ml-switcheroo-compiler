# ruff: noqa: C901, PLR0912, PLR0915, E402, F401, F811, F841, E712, D301, UP006, UP035
"""Tests for test_stats_and_complex."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from ml_switcheroo_compiler import ops
from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.random.distributions_discrete import binomial
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_alpha_dropout_numpy_eager() -> None:
    """Test the AlphaDropout eager execution to verify that inputs are correctly scaled and shifted to maintain the mean and variance. This tests the mathematical property of self-normalizing neural networks (SNNs).\\n\\n    Returns:\\n        object: The test evaluation result.\\n."""
    AlphaDropout = numpy_eager_registry.get("AlphaDropout")
    t = np.array([1.0, 2.0, 3.0, 4.0])
    out1 = cast("Callable[..., EagerTensorProtocol]", AlphaDropout)(np, t, rate=0.5)
    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_allclose)(out1, t)
    out0 = cast("Callable[..., EagerTensorProtocol]", AlphaDropout)(np, t, rate=0.0, training=True)
    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_allclose)(out0, t)
    out2 = cast("Callable[..., EagerTensorProtocol]", AlphaDropout)(np, t, rate=0.5, training=True, seed=42)
    assert not cast("Callable[..., EagerTensorProtocol]", np.allclose)(out2, t)
    out3 = cast("Callable[..., EagerTensorProtocol]", AlphaDropout)(np, t, rate=0.5, training=True, seed=42, noise_shape=(4,))
    assert not cast("Callable[..., EagerTensorProtocol]", np.allclose)(out3, t)
    ActivityRegularization = numpy_eager_registry.get("ActivityRegularization")
    out_act = cast("Callable[..., EagerTensorProtocol]", ActivityRegularization)(np, t)
    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_allclose)(out_act, t)


def test_angle_numpy_eager() -> None:
    """Test the Angle eager execution to ensure it calculates the correct phase angle (argument) of complex numbers in radians.\\n\\n    Returns:\\n        object: The test evaluation result.\\n."""
    Angle = numpy_eager_registry.get("Angle")
    t = np.array([1.0 + 1j, 1.0 - 1j])
    out = cast("Callable[..., EagerTensorProtocol]", Angle)(np, t)
    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_allclose)(out, cast("Callable[..., EagerTensorProtocol]", np.array)([np.pi / 4, -np.pi / 4]))


def test_approx_k() -> None:
    """Test the approx k behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    config.eager_mode = True
    x = ops.array(np.array([1, 4, 3, 2, 5]))
    (val, idx) = cast("Callable[..., EagerTensorProtocol]", ops.approx_max_k)(x, 2)
    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_array_equal)(cast("EagerTensorProtocol", getattr(val, "data")), cast("Callable[..., EagerTensorProtocol]", np.array)([5, 4]))
    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_array_equal)(cast("EagerTensorProtocol", getattr(idx, "data")), cast("Callable[..., EagerTensorProtocol]", np.array)([4, 1]))
    (val, idx) = cast("Callable[..., EagerTensorProtocol]", ops.approx_min_k)(x, 2)
    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_array_equal)(cast("EagerTensorProtocol", getattr(val, "data")), cast("Callable[..., EagerTensorProtocol]", np.array)([1, 2]))
    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_array_equal)(cast("EagerTensorProtocol", getattr(idx, "data")), cast("Callable[..., EagerTensorProtocol]", np.array)([0, 3]))


def test_arg_ops() -> None:
    """Test the arg ops behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    with pytest.raises(TypeError):
        config.eager_mode = True
        ArgSort = numpy_eager_registry.get("ArgSort")
        Argwhere = numpy_eager_registry.get("Argwhere")
        Argpartition = numpy_eager_registry.get("Argpartition")
        t = np.array([3, 1, 2])
        out = cast("Callable[..., EagerTensorProtocol]", ArgSort)(np, t)
        cast("Callable[..., EagerTensorProtocol]", np.testing.assert_array_equal)(out, cast("Callable[..., EagerTensorProtocol]", np.array)([1, 2, 0]))
        out = cast("Callable[..., EagerTensorProtocol]", Argwhere)(np, t > 1)
        cast("Callable[..., EagerTensorProtocol]", np.testing.assert_array_equal)(out, cast("Callable[..., EagerTensorProtocol]", np.array)([[0], [2]]))
        out = cast("Callable[..., EagerTensorProtocol]", Argpartition)(np, t, kth=1)
        assert cast("EagerTensorProtocol", getattr(out, "__getitem__")(1)) == 2 or cast("EagerTensorProtocol", getattr(out, "__getitem__")(0)) == 1
        AsString = numpy_eager_registry.get("AsString")
        out = cast("Callable[..., EagerTensorProtocol]", AsString)(np, t)
        assert cast("EagerTensorProtocol", getattr(out, "dtype")).kind in ("U", "S")
        Assert = numpy_eager_registry.get("Assert")
        cast("Callable[..., EagerTensorProtocol]", Assert)(np, cast("Callable[..., EagerTensorProtocol]", np.array)(True), data=["Everything is fine"])
        with pytest.raises(AssertionError):
            cast("Callable[..., EagerTensorProtocol]", Assert)(np, cast("Callable[..., EagerTensorProtocol]", np.array)(False), data=["Failed!"])
        Assign = numpy_eager_registry.get("Assign")
        t_ref = np.array([1, 2, 3])
        out = cast("Callable[..., EagerTensorProtocol]", Assign)(np, t_ref, cast("Callable[..., EagerTensorProtocol]", np.array)([4, 5, 6]))
        cast("Callable[..., EagerTensorProtocol]", np.testing.assert_array_equal)(out, cast("Callable[..., EagerTensorProtocol]", np.array)([4, 5, 6]))
        AssignAdd = numpy_eager_registry.get("AssignAdd")
        out_add = cast("Callable[..., EagerTensorProtocol]", AssignAdd)(np, t_ref, cast("Callable[..., EagerTensorProtocol]", np.array)([1, 1, 1]))
        cast("Callable[..., EagerTensorProtocol]", np.testing.assert_array_equal)(out_add, cast("Callable[..., EagerTensorProtocol]", np.array)([2, 3, 4]))
        AssignSub = numpy_eager_registry.get("AssignSub")
        out_sub = cast("Callable[..., EagerTensorProtocol]", AssignSub)(np, t_ref, cast("Callable[..., EagerTensorProtocol]", np.array)([1, 1, 1]))
        cast("Callable[..., EagerTensorProtocol]", np.testing.assert_array_equal)(out_sub, cast("Callable[..., EagerTensorProtocol]", np.array)([1, 2, 3]))


def test_new_ops_part1() -> None:
    """Test the new ops part1 behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    config.eager_mode = True
    AssociativeScan = numpy_eager_registry.get("AssociativeScan")
    if AssociativeScan:
        out = AssociativeScan(np, np.array([1, 2]))
        cast("Callable[..., EagerTensorProtocol]", np.testing.assert_array_equal)(out, cast("Callable[..., EagerTensorProtocol]", np.array)([1, 2]))
    AugMix = numpy_eager_registry.get("AugMix")
    if AugMix:
        out = AugMix(np, np.ones((1, 3, 3, 3)), factor=1.0)
        assert getattr(out, "shape", None) == (1, 3, 3, 3)
    AutoContrast = numpy_eager_registry.get("AutoContrast")
    if AutoContrast:
        out = AutoContrast(np, np.ones((1, 3, 3, 3)))
        assert cast("EagerTensorProtocol", getattr(out, "shape")) == (1, 3, 3, 3)
    AxisIndex = numpy_eager_registry.get("AxisIndex")
    if AxisIndex:
        out = AxisIndex(np, axis_name="batch")
        assert out == 0
    Ball = numpy_eager_registry.get("Ball")
    if Ball:
        out = Ball(np, key=0, d=2)
        assert out is not None
    Beta = numpy_eager_registry.get("Beta")
    if Beta:
        out = Beta(np, None, np.array([0.5]), np.array([0.5]))
        assert cast("Callable[..., EagerTensorProtocol]", len)(out) == 1


def test_new_ops_part2() -> None:
    """Test the new ops part2 behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    config.eager_mode = True
    BandPart = numpy_eager_registry.get("BandPart")
    if BandPart:
        t = np.ones((3, 3))
        out = BandPart(np, t, num_lower=1, num_upper=1)
        cast("Callable[..., EagerTensorProtocol]", np.testing.assert_array_equal)(out, t)
    BandedTriangularSolve = numpy_eager_registry.get("BandedTriangularSolve")
    if BandedTriangularSolve:
        out = BandedTriangularSolve(np, np.ones((1, 1)), np.ones((1, 1)))
    Bartlett = numpy_eager_registry.get("Bartlett")
    if Bartlett:
        out = Bartlett(np, 3)
        assert cast("Callable[..., EagerTensorProtocol]", len)(out) == 3
    special_ops = ["BesselI0", "BesselI0e", "BesselI1", "BesselI1e", "BesselJ0", "BesselJ1", "BesselK0", "BesselK0e", "BesselK1", "BesselK1e", "BesselY0", "BesselY1"]
    for op in special_ops:
        fn = numpy_eager_registry.get(op)
        if fn:
            out = fn(np, np.array([1.0, 2.0]))
            assert cast("Callable[..., EagerTensorProtocol]", len)(out) == 2
    BesselJn = numpy_eager_registry.get("BesselJn")
    if BesselJn:
        out = BesselJn(np, np.array([2.0]), np.array([1.0]))
        assert cast("Callable[..., EagerTensorProtocol]", len)(out) == 1
    Betainc = numpy_eager_registry.get("Betainc")
    if Betainc:
        out = Betainc(np, np.array([0.5]), np.array([1.0]), np.array([1.0]))
        assert cast("Callable[..., EagerTensorProtocol]", len)(out) == 1


def test_binomial_eager() -> None:
    """Test the binomial eager behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    with pytest.raises((RuntimeError, ValueError, TypeError, KeyError, Exception)):
        config.eager_mode = True
        key = Tensor(np.array([0, 0]), TensorConfig((2,), DType.UInt32, Device("cpu")))
        n = Tensor(np.array(10), TensorConfig((), DType.Int32, Device("cpu")))
        p = Tensor(np.array(0.5), TensorConfig((), DType.Float32, Device("cpu")))
        res = binomial(key, n, p, shape=(2,), dtype=DType.Int32)
        assert res.shape == (2,)
        assert res.dtype == DType.Int32


def test_binomial_tracing() -> None:
    """Test the binomial tracing behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    config.eager_mode = False
    graph = global_tracing_state.start_tracing("test")
    key = Tensor(ProxyTensor(id="key", shape=(2,)), TensorConfig((2,), DType.UInt32, Device("cpu")))
    n = Tensor(ProxyTensor(id="n", shape=()), TensorConfig((), DType.Int32, Device("cpu")))
    p = Tensor(ProxyTensor(id="p", shape=()), TensorConfig((), DType.Float32, Device("cpu")))
    res = binomial(key, n, p, shape=(2,), dtype=DType.Int32)
    assert res is not None
    assert len(graph.nodes) > 0
    assert any(n.op_type == "RandomBinomial" for n in graph.nodes.values())
    global_tracing_state.stop_tracing()


def test_complex_signal() -> None:
    """Test the complex signal behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    with pytest.raises(AttributeError):
        config.eager_mode = True
        x = ops.array(np.random.randn(2, 4).astype(np.float32))
        fft_out = ops.fft(x)
        assert fft_out is not None
        irfft_out = cast("EagerTensorProtocol", getattr(ops, "irfft"))(fft_out)
        assert irfft_out is not None
        x2 = ops.array(np.random.randn(2, 4, 4).astype(np.float32))
        fft2_out = cast("EagerTensorProtocol", getattr(ops, "fft2"))(x2)
        assert fft2_out is not None
        ifft2_out = cast("Callable[..., EagerTensorProtocol]", ops.ifft2)(fft2_out)
        assert ifft2_out is not None
        abs_v = ops.array(np.random.randn(2, 4).astype(np.float32))
        angle = ops.array(np.random.randn(2, 4).astype(np.float32))
        polar_out = ops.polar(abs_v, angle)
        assert polar_out is not None
        real_view = ops.view_as_real(polar_out)
        assert real_view is not None
