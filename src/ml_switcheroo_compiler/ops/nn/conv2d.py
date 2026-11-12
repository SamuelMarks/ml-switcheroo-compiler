"""Convolution operations."""

from __future__ import annotations

import typing  # noqa: F401
from collections.abc import Sequence

import pydantic

from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.core.tensor import Tensor, ensure_tensor
from ml_switcheroo_compiler.ops.configs import ConvConfig
from ml_switcheroo_compiler.ops.linalg import conv_general_dilated
from ml_switcheroo_compiler.ops.registry import get_op

from .conv1d import _ConvHyperparamsProtocol
from .conv_utils import _build_conv_config, _prepare_depthwise_conv


class ConvHyperparams(pydantic.BaseModel):
    """Conv hyperparameters."""

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    strides: Sequence[int] | int = 1
    padding: str | Sequence[tuple[int, int]] = "VALID"


def conv2d(lhs: Tensor, rhs: Tensor, config_obj: ConvConfig | TensorOrScalar = None, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """2D Convolution.

    Args:
        lhs (Tensor): Input `lhs` argument (lhs).
        rhs (Tensor): Input `rhs` argument (rhs).
        config_obj (TensorOrScalar): Input `config_obj` argument (config obj).
        **kwargs (TensorOrScalar): Keyword args.

    Returns:
        Tensor: Computed `Tensor` output.
    """
    if config_obj is None:
        config_obj = _build_conv_config(kwargs, ((0, 3, 1, 2), (3, 2, 0, 1), (0, 3, 1, 2)))
    assert isinstance(config_obj, ConvConfig)

    return conv_general_dilated(lhs, rhs, config_obj)


def conv2d_transpose(
    lhs: Tensor,
    rhs: Tensor,
    strides: Sequence[int] | int = 1,
    padding: str | Sequence[tuple[int, int]] = "VALID",
) -> TensorOrScalar:
    """2D convolution transpose.

    Args:
        lhs (Tensor): Input `lhs` argument (lhs).
        rhs (Tensor): Input `rhs` argument (rhs).
        strides (Union): Input `strides` argument (strides).
        padding (Union): Input `padding` argument (padding).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    conv_transpose = get_op("ConvTranspose")()

    return conv_transpose(lhs, rhs, strides, padding)


def depthwise_conv2d(lhs: Tensor, rhs: Tensor, config_obj: ConvConfig | TensorOrScalar = None, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """2D Depthwise Convolution.

    Args:
        lhs (Tensor): Input `lhs` argument (lhs).
        rhs (Tensor): Input `rhs` argument (rhs).
        config_obj (TensorOrScalar): Input `config_obj` argument (config obj).
        **kwargs (TensorOrScalar): Keyword args.

    Returns:
        Tensor: Computed `Tensor` output.
    """
    dimension_numbers = ((0, 3, 1, 2), (3, 2, 0, 1), (0, 3, 1, 2))
    rhs_reshaped, config_obj = _prepare_depthwise_conv(lhs, rhs, 2, dimension_numbers, config_obj, **kwargs)
    assert isinstance(config_obj, ConvConfig)
    return conv_general_dilated(lhs, ensure_tensor(rhs_reshaped), config_obj)


def separable_conv2d(
    lhs: Tensor,
    depthwise_filter: Tensor,
    pointwise_filter: Tensor,
    config: ConvHyperparams | TensorOrScalar = None,
    **kwargs: TensorOrScalar,
) -> TensorOrScalar:
    """2D Separable Convolution.

    Args:
        lhs (Tensor): Input `lhs` argument (lhs).
        depthwise_filter (Tensor): Input `depthwise_filter` argument (depthwise filter).
        pointwise_filter (Tensor): Input `pointwise_filter` argument (pointwise filter).
        config (ConvHyperparams): Input `config` argument (config).
        **kwargs (TensorOrScalar): Keyword args.

    Returns:
        Tensor: Computed `Tensor` output.
    """
    cfg = config if isinstance(config, (ConvHyperparams, _ConvHyperparamsProtocol)) else ConvHyperparams()
    kwargs["strides"] = cfg.strides
    kwargs["padding"] = cfg.padding
    depthwise_out = depthwise_conv2d(lhs, depthwise_filter, None, **kwargs)
    return conv2d(ensure_tensor(depthwise_out), pointwise_filter, None, strides=1, padding="VALID")
