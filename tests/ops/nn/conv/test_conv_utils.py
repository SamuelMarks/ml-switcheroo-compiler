"""Tests for `ml_switcheroo_compiler.ops.nn.conv_utils`."""

from __future__ import annotations

import typing
from unittest.mock import patch

import numpy as np

import ml_switcheroo_compiler.tracing.state as state
from ml_switcheroo_compiler.core.config import ConfigContext, config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.nn.conv_utils import (
    _build_conv_config,
    _calc_same_pad,
    _calc_valid_pad,
    _calculate_conv_transpose_padding,
    _prepare_depthwise_conv,
    atrous_conv2d_transpose,
    bias_add,
    collapse_repeated,
    compute_average_loss,
    conv_transpose,
    depthwise_conv2d_backprop_filter,
    depthwise_conv2d_backprop_input,
    dilation2d,
    erosion2d,
)

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


def test_conv_utils_extra() -> None:
    """Verify padding helpers in `conv_utils`."""
    assert _calc_same_pad(k=2, s=3) == (1, 2)
    assert _calc_same_pad(k=2, s=1) == (1, 0)
    assert _calc_valid_pad(k=2, s=1) == (1, 1)
    assert _calculate_conv_transpose_padding("VALID", (2,), (1,)) == [(1, 1)]
    assert _calculate_conv_transpose_padding("SAME", (2,), (1,)) == [(1, 0)]
    assert _calculate_conv_transpose_padding([(1, 1)], (2,), (1,)) == [(1, 1)]


def test_prepare_depthwise_conv_extra() -> None:
    """Verify `_prepare_depthwise_conv` reshapes filters and populates dilation settings."""
    device = Device("cpu")
    t1 = Tensor(np.ones((1, 5, 2)), TensorConfig((1, 5, 2), "float32", device))
    t2 = Tensor(np.ones((3, 2, 4)), TensorConfig((3, 2, 4), "float32", device))
    with ConfigContext(eager_mode=True):
        with patch("ml_switcheroo_compiler.ops.nn.conv_utils.reshape") as mock_reshape:
            mock_reshape.return_value = "reshaped"
            rhs_reshaped, conf = typing.cast("EagerCallableProtocol", _prepare_depthwise_conv)(t1, t2, 1, ((0, 1, 2), (0, 1, 2), (0, 1, 2)), strides=1, lhs_dilation=1, rhs_dilation=1)
            assert rhs_reshaped == "reshaped"
            assert typing.cast("EagerTensorProtocol", getattr(conf, "lhs_dilation")) == (1,)
            assert typing.cast("EagerTensorProtocol", getattr(conf, "rhs_dilation")) == (1,)


def create_eager_tensor(data: EagerValue) -> EagerValue:
    """Create a CPU float32 Tensor from a NumPy array.

    Args:
        data: Test parameter `data`.
    """
    return typing.cast("EagerCallableProtocol", Tensor)(
        data,
        typing.cast("EagerCallableProtocol", TensorConfig)(
            typing.cast("EagerTensorProtocol", getattr(data, "shape")),
            DType.Float32,
            typing.cast("EagerCallableProtocol", Device)("cpu"),
        ),
    )


def test_conv_utils_missing_branches() -> None:
    """Verify `_build_conv_config` and `_prepare_depthwise_conv` configuration branches."""
    dim_nums = ((0, 3, 1, 2), (3, 2, 0, 1), (0, 3, 1, 2))

    cfg = _build_conv_config({"strides": (1, 1), "lhs_dilation": (1, 1), "rhs_dilation": (1, 1)}, dim_nums)
    assert getattr(cfg, "window_strides", None) == (1, 1)

    cfg_ints = _build_conv_config({"strides": 1, "lhs_dilation": 1, "rhs_dilation": 1}, dim_nums)
    assert getattr(cfg_ints, "window_strides", None) == (1, 1)
    assert getattr(cfg_ints, "lhs_dilation", None) == (1, 1)
    assert getattr(cfg_ints, "rhs_dilation", None) == (1, 1)

    with ConfigContext(eager_mode=True):
        lhs = create_eager_tensor(np.ones((1, 2, 2, 3)))
        rhs = create_eager_tensor(np.ones((2, 2, 3, 1)))

        r, c = typing.cast("EagerCallableProtocol", _prepare_depthwise_conv)(lhs, rhs, 2, dim_nums, config_obj="mock_config")
        assert typing.cast("EagerTensorProtocol", getattr(r, "shape")) == (2, 2, 1, 3)
        assert c == "mock_config"

        r2, c2 = typing.cast("EagerCallableProtocol", _prepare_depthwise_conv)(lhs, rhs, 2, dim_nums, strides=(1, 1), lhs_dilation=(1, 1), rhs_dilation=(1, 1))
        assert typing.cast("EagerTensorProtocol", getattr(r2, "shape")) == (2, 2, 1, 3)
        assert getattr(c2, "window_strides", None) == (1, 1)


def test_conv_utils_eager_and_trace_coverage() -> None:
    """Verify eager and tracing paths for bias_add, collapse_repeated, dilation2d, erosion2d, and conv_transpose."""

    class DummyBackend:
        """Stub backend for convolution dispatch testing."""

        def execute_op(self, op: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Return op name or dummy array.

            Args:
                op: Test parameter `op`.
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            _ = (args, kwargs)
            return "dummy"

    t = Tensor(np.array([1]), TensorConfig((1,), "int32", "cpu"))

    config.eager_mode = True
    res_bias = typing.cast(Tensor, bias_add(t, t))
    np.testing.assert_allclose(np.asarray(res_bias.data), np.array([2]))

    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=DummyBackend()):
        assert collapse_repeated(1, 2) == "dummy"
        assert dilation2d(1, 2) == "dummy"
        assert erosion2d(1, 2) == "dummy"
        assert conv_transpose(1, 2, (1, 2)) == "dummy"

        res_filt_eager = typing.cast(Tensor, depthwise_conv2d_backprop_filter(t, (3, 3, 1, 1), t))
        assert res_filt_eager.shape == (3, 3, 1, 1)
        res_filt_int = typing.cast(Tensor, depthwise_conv2d_backprop_filter(t, 4, t))
        assert res_filt_int.shape == (4,)
        res_filt_t = typing.cast(Tensor, depthwise_conv2d_backprop_filter(t, t, t))
        assert res_filt_t.shape == (1,)

        res_inp_eager = typing.cast(Tensor, depthwise_conv2d_backprop_input((1, 4, 4, 1), t, t))
        assert res_inp_eager.shape == (1, 4, 4, 1)
        res_inp_int = typing.cast(Tensor, depthwise_conv2d_backprop_input(4, t, t))
        assert res_inp_int.shape == (4,)
        res_inp_t = typing.cast(Tensor, depthwise_conv2d_backprop_input(t, t, t))
        assert res_inp_t.shape == (1,)

    config.eager_mode = False
    state.global_tracing_state.start_tracing("test_conv_utils_trace")
    try:
        res_col, res_seq = typing.cast("tuple[Tensor, Tensor]", collapse_repeated(t, t))
        assert res_col.shape == (1,)
        assert res_seq.shape == (1,)
        res_ct = typing.cast(Tensor, conv_transpose(t, t, (1, 2)))
        assert res_ct.shape == (1, 2)
        res_dil = typing.cast(Tensor, dilation2d(t, t))
        assert res_dil.shape == (1,)
        res_ero = typing.cast(Tensor, erosion2d(t, t))
        assert res_ero.shape == (1,)
        res_filt_trace = typing.cast(Tensor, depthwise_conv2d_backprop_filter(t, (3, 3, 1, 1), t))
        assert res_filt_trace.shape == (3, 3, 1, 1)
        res_inp_trace = typing.cast(Tensor, depthwise_conv2d_backprop_input((1, 4, 4, 1), t, t))
        assert res_inp_trace.shape == (1, 4, 4, 1)
    finally:
        state.global_tracing_state.stop_tracing()
        config.eager_mode = True


def test_conv_utils_missing_others() -> None:
    """Verify `atrous_conv2d`, `atrous_conv2d_transpose`, `depthwise_conv2d`, `convolution`, and `compute_average_loss`."""
    from ml_switcheroo_compiler.ops.nn.conv_utils import atrous_conv2d, convolution, depthwise_conv2d

    orig = config.eager_mode
    config.eager_mode = True
    try:
        t_loss = Tensor(np.array([2.0, 4.0], dtype=np.float32), TensorConfig((2,), "float32", "cpu"))
        t_weight = Tensor(np.array([0.5, 0.5], dtype=np.float32), TensorConfig((2,), "float32", "cpu"))
        avg1 = typing.cast(Tensor, compute_average_loss(t_loss))
        np.testing.assert_allclose(np.asarray(avg1.data), 3.0)
        avg2 = typing.cast(Tensor, compute_average_loss(t_loss, sample_weight=t_weight, global_batch_size=4))
        np.testing.assert_allclose(np.asarray(avg2.data), 1.5)

        t_4d = Tensor(np.ones((1, 2, 2, 1), dtype=np.float32), TensorConfig((1, 2, 2, 1), "float32", "cpu"))
        with patch("ml_switcheroo_compiler.ops.nn.conv_utils.get_op") as mock_get_op:
            mock_get_op.return_value = lambda: lambda *a, **kw: "dispatched_op"
            assert atrous_conv2d(t_4d, t_4d, 2, "SAME") == "dispatched_op"
            assert atrous_conv2d_transpose(t_4d, t_4d, (1, 2, 2, 1)) == "dispatched_op"
            assert depthwise_conv2d(t_4d, t_4d) == "dispatched_op"
            assert convolution(t_4d, t_4d) == "dispatched_op"
    finally:
        config.eager_mode = orig
