"""Convolution operations."""

import typing
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Union

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.linalg import conv_general_dilated
from ml_switcheroo_compiler.ops.registry import get_op

from .conv_utils import _build_conv_config, _prepare_depthwise_conv


@dataclass
class ConvHyperparams:
    """Conv hyperparameters."""

    strides: Union[Sequence[int], int] = 1
    padding: Union[str, Sequence[tuple[int, int]]] = "VALID"


def conv2d(lhs: Tensor, rhs: Tensor, config_obj: typing.Optional[object] = None, **kwargs: object) -> Tensor:
    """2D Convolution.

    Args:
        lhs (Tensor): Left-hand side tensor (batch, height, width, in_channels).
        rhs (Tensor): Right-hand side tensor (height, width, in_channels, out_channels).
        config_obj (ConvConfig | None): Configuration.

        kwargs (object): Additional kwargs.\

    Returns:
        Tensor: The result of the convolution.
    """
    if config_obj is None:
        config_obj = _build_conv_config(kwargs, ((0, 3, 1, 2), (3, 2, 0, 1), (0, 3, 1, 2)))

    return conv_general_dilated(lhs, rhs, config_obj)


def conv2d_transpose(
    lhs: Tensor,
    rhs: Tensor,
    strides: Union[Sequence[int], int] = 1,
    padding: Union[str, Sequence[tuple[int, int]]] = "VALID",
) -> Tensor:
    """2D convolution transpose.

    Args:
        lhs (Tensor): Left-hand side tensor.
        rhs (Tensor): Right-hand side tensor.
        strides (Union[Sequence[int], int]): Strides.
        padding (Union[str, Sequence[tuple[int, int]]]): Padding.

        kwargs (object): Additional kwargs.\

    Returns:
        Tensor: The result of the convolution.
    """
    conv_transpose = get_op("ConvTranspose")()

    return conv_transpose(lhs, rhs, strides, padding)


def depthwise_conv2d(lhs: Tensor, rhs: Tensor, config_obj: typing.Optional[object] = None, **kwargs: object) -> Tensor:
    """2D Depthwise Convolution.

    Args:
        lhs (Tensor): Left-hand side tensor (batch, height, width, in_channels).
        rhs (Tensor): Right-hand side tensor (height, width, in_channels, channel_multiplier).
        config_obj (ConvConfig | None): Configuration.

        kwargs (object): Additional kwargs.\

    Returns:
        Tensor: The result of the convolution.
    """
    dimension_numbers = ((0, 3, 1, 2), (3, 2, 0, 1), (0, 3, 1, 2))
    rhs_reshaped, config_obj = _prepare_depthwise_conv(lhs, rhs, 2, dimension_numbers, config_obj, **kwargs)
    return conv_general_dilated(lhs, rhs_reshaped, config_obj)


def separable_conv2d(
    lhs: Tensor,
    depthwise_filter: Tensor,
    pointwise_filter: Tensor,
    config: ConvHyperparams = None,
    **kwargs: object,
) -> Tensor:
    """2D Separable Convolution.

    Args:
        lhs (Tensor): Left-hand side tensor.
        depthwise_filter (Tensor): Depthwise filter.
        pointwise_filter (Tensor): Pointwise filter.
        config (ConvHyperparams): Hyperparameters.
        kwargs (object): kwargs.
    """
    config = config or ConvHyperparams()
    strides, padding = config.strides, config.padding
    kwargs["strides"] = strides
    kwargs["padding"] = padding
    depthwise_out = depthwise_conv2d(lhs, depthwise_filter, None, **kwargs)
    return conv2d(depthwise_out, pointwise_filter, None, strides=1, padding="VALID")
