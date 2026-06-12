"""Docstring module."""

import ml_switcheroo.nn as nn

# ruff: noqa: F403, F405
"""Tests for Neural Network primitives."""

import pytest  # noqa: E402
import numpy as np  # noqa: E402
from ml_switcheroo.core import (  # noqa: E402
    ConfigContext,
    DType,
    Tensor,
    Device,
    DeviceType,
    UnimplementedMathError,
)
from ml_switcheroo.tracing import _tracer, ProxyTensor  # noqa: E402
from ml_switcheroo.nn import relu, conv1d, lstm_cell  # noqa: E402


def _make_tensor(data_list: object, dtype: object, shape: object = None) -> object:
    """Docstring."""
    if shape is None:
        shape = (len(data_list),)
    return Tensor(
        np.array(data_list, dtype=dtype.value).reshape(shape),
        shape,
        dtype,
        Device(DeviceType.CPU),
    )


def _make_proxy(shape: object, dtype: object) -> object:
    """Docstring."""
    return Tensor(
        ProxyTensor("a", shape, dtype.value), shape, dtype, Device(DeviceType.CPU)
    )


def test_activations_eager() -> None:
    """Docstring."""
    with ConfigContext(eager_mode=True):
        t = _make_tensor([1.0, -1.0], DType.Float32, (2,))

        for op in ["relu", "leaky_relu", "swish", "sigmoid", "tanh", "elu", "celu"]:
            res = getattr(nn, op)(t)
            assert isinstance(res, Tensor)

        for op in [
            "gelu",
            "softplus",
            "glu",
            "selu",
            "mish",
            "hardswish",
            "softmax",
            "log_softmax",
        ]:
            with pytest.raises(UnimplementedMathError):
                getattr(nn, op)(t)


def test_activations_tracing() -> None:
    """Docstring."""
    with ConfigContext(eager_mode=False):
        t = _make_proxy((2,), DType.Float32)

        with pytest.raises(RuntimeError):
            relu(t)

        _ = _tracer.start_tracing()
        try:
            for op in [
                "relu",
                "leaky_relu",
                "gelu",
                "swish",
                "sigmoid",
                "tanh",
                "softplus",
                "elu",
                "selu",
                "celu",
                "glu",
                "mish",
                "hardswish",
                "softmax",
                "log_softmax",
            ]:
                res = getattr(nn, op)(t)
                assert isinstance(res, Tensor)
        finally:
            _tracer.stop_tracing()


def test_complex_eager() -> None:
    """Docstring."""
    with ConfigContext(eager_mode=True):
        t = _make_tensor([1.0, -1.0], DType.Float32, (2,))
        for op in [
            "conv1d",
            "conv2d",
            "conv3d",
            "conv_transpose1d",
            "conv_transpose2d",
            "conv_transpose3d",
            "max_pool1d",
            "max_pool2d",
            "max_pool3d",
            "avg_pool1d",
            "avg_pool2d",
            "avg_pool3d",
            "adaptive_avg_pool2d",
            "fractional_max_pool2d",
            "layer_norm",
            "batch_norm",
            "group_norm",
            "rms_norm",
            "instance_norm",
            "dropout",
            "alpha_dropout",
            "feature_alpha_dropout",
            "spatial_dropout",
            "embedding",
            "pad",
            "upsample_bilinear",
            "upsample_nearest",
            "scaled_dot_product_attention",
            "rnn_cell",
            "gru_cell",
        ]:
            with pytest.raises(UnimplementedMathError):
                if op == "conv1d":
                    getattr(nn, op)(t, t)
                elif op == "layer_norm":
                    getattr(nn, op)(t, (2,))
                elif op == "batch_norm":
                    getattr(nn, op)(t, t, t)
                elif op == "group_norm":
                    getattr(nn, op)(t, 1)
                elif op == "rms_norm":
                    getattr(nn, op)(t, (2,))
                elif op == "adaptive_avg_pool2d":
                    getattr(nn, op)(t, (1, 1))
                elif op == "fractional_max_pool2d":
                    getattr(nn, op)(t, 2)
                elif op == "embedding":
                    getattr(nn, op)(t, t)
                elif op == "pad":
                    getattr(nn, op)(t, (1, 1))
                elif op == "scaled_dot_product_attention":
                    getattr(nn, op)(t, t, t)
                elif op == "rnn_cell":
                    getattr(nn, op)(t, t, t, t)
                elif op == "gru_cell":
                    getattr(nn, op)(t, t, t, t)
                else:
                    getattr(nn, op)(t, 2)

        with pytest.raises(UnimplementedMathError):
            lstm_cell(t, (t, t), t, t)


def test_complex_tracing() -> None:
    """Docstring."""
    with ConfigContext(eager_mode=False):
        t = _make_proxy((2,), DType.Float32)

        with pytest.raises(RuntimeError):
            conv1d(t, t)

        _ = _tracer.start_tracing()
        try:
            for op in [
                "conv1d",
                "conv2d",
                "conv3d",
                "conv_transpose1d",
                "conv_transpose2d",
                "conv_transpose3d",
                "max_pool1d",
                "max_pool2d",
                "max_pool3d",
                "avg_pool1d",
                "avg_pool2d",
                "avg_pool3d",
                "adaptive_avg_pool2d",
                "fractional_max_pool2d",
                "layer_norm",
                "batch_norm",
                "group_norm",
                "rms_norm",
                "instance_norm",
                "dropout",
                "alpha_dropout",
                "feature_alpha_dropout",
                "spatial_dropout",
                "embedding",
                "pad",
                "upsample_bilinear",
                "upsample_nearest",
                "scaled_dot_product_attention",
                "rnn_cell",
                "gru_cell",
            ]:
                if op == "conv1d":
                    getattr(nn, op)(t, t)
                elif op == "layer_norm":
                    getattr(nn, op)(t, (2,))
                elif op == "batch_norm":
                    getattr(nn, op)(t, t, t)
                elif op == "group_norm":
                    getattr(nn, op)(t, 1)
                elif op == "rms_norm":
                    getattr(nn, op)(t, (2,))
                elif op == "adaptive_avg_pool2d":
                    getattr(nn, op)(t, (1, 1))
                elif op == "fractional_max_pool2d":
                    getattr(nn, op)(t, 2)
                elif op == "embedding":
                    getattr(nn, op)(t, t)
                elif op == "pad":
                    getattr(nn, op)(t, (1, 1))
                elif op == "scaled_dot_product_attention":
                    getattr(nn, op)(t, t, t)
                elif op == "rnn_cell":
                    getattr(nn, op)(t, t, t, t)
                elif op == "gru_cell":
                    getattr(nn, op)(t, t, t, t)
                else:
                    getattr(nn, op)(t, 2)

            res = lstm_cell(t, (t, t), t, t, t, t)
            assert isinstance(res, tuple)
            assert len(res) == 2
        finally:
            _tracer.stop_tracing()
