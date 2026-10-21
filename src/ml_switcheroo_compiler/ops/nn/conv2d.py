"""Convolution operations."""

from __future__ import annotations

import typing  # noqa: F401
from collections.abc import Sequence

import pydantic

from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.core.tensor import Tensor, as_tensor_obj
from ml_switcheroo_compiler.ops.configs import ConvConfig
from ml_switcheroo_compiler.ops.linalg import conv_general_dilated
from ml_switcheroo_compiler.ops.registry import get_op

from .conv_utils import _build_conv_config, _prepare_depthwise_conv


class ConvHyperparams(pydantic.BaseModel):
    """Conv hyperparameters."""

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    strides: Sequence[int] | int = 1
    padding: str | Sequence[tuple[int, int]] = "VALID"


def conv2d(lhs: Tensor, rhs: Tensor, config_obj: TensorOrScalar = None, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """2D Convolution.

    Args:
        lhs (Tensor): The lhs parameter.
        rhs (Tensor): The rhs parameter.
        config_obj (TensorOrScalar): The config_obj parameter.
        **kwargs (TensorOrScalar): Keyword args.

    Returns:
        Tensor: Result.
    """
    if config_obj is None:
        config_obj = _build_conv_config(kwargs, ((0, 3, 1, 2), (3, 2, 0, 1), (0, 3, 1, 2)))

    return conv_general_dilated(lhs, rhs, typing.cast(ConvConfig, config_obj))


def conv2d_transpose(
    lhs: Tensor,
    rhs: Tensor,
    strides: Sequence[int] | int = 1,
    padding: str | Sequence[tuple[int, int]] = "VALID",
) -> TensorOrScalar:
    """2D convolution transpose.

    Args:
        lhs (Tensor): The lhs parameter.
        rhs (Tensor): The rhs parameter.
        strides (Union): The strides parameter.
        padding (Union): The padding parameter.

    Returns:
        Tensor: Result.
    """
    conv_transpose = get_op("ConvTranspose")()

    return conv_transpose(lhs, rhs, strides, padding)


def depthwise_conv2d(lhs: Tensor, rhs: Tensor, config_obj: TensorOrScalar = None, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """2D Depthwise Convolution.

    Args:
        lhs (Tensor): The lhs parameter.
        rhs (Tensor): The rhs parameter.
        config_obj (TensorOrScalar): The config_obj parameter.
        **kwargs (TensorOrScalar): Keyword args.

    Returns:
        Tensor: Result.
    """
    dimension_numbers = ((0, 3, 1, 2), (3, 2, 0, 1), (0, 3, 1, 2))
    rhs_reshaped, config_obj = getattr(_prepare_depthwise_conv(lhs, rhs, 2, dimension_numbers, config_obj, **kwargs), "__iter__")()
    return conv_general_dilated(lhs, rhs_reshaped, typing.cast(ConvConfig, config_obj))


def separable_conv2d(
    lhs: Tensor,
    depthwise_filter: Tensor,
    pointwise_filter: Tensor,
    config: TensorOrScalar = None,
    **kwargs: TensorOrScalar,
) -> TensorOrScalar:
    """2D Separable Convolution.

    Args:
        lhs (Tensor): The lhs parameter.
        depthwise_filter (Tensor): The depthwise_filter parameter.
        pointwise_filter (Tensor): The pointwise_filter parameter.
        config (ConvHyperparams): The config parameter.
        **kwargs (TensorOrScalar): Keyword args.

    Returns:
        Tensor: Result.
    """
    config = config or ConvHyperparams()
    strides, padding = getattr(config, "strides"), getattr(config, "padding")
    kwargs["strides"] = strides
    kwargs["padding"] = padding
    depthwise_out = depthwise_conv2d(lhs, depthwise_filter, None, **kwargs)
    return conv2d(as_tensor_obj(depthwise_out), pointwise_filter, None, strides=1, padding="VALID")
