# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Convolution operations."""

import typing
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Union

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.linalg import conv_general_dilated
from ml_switcheroo_compiler.ops.registry import get_op

from .conv_utils import _build_conv_config, _prepare_depthwise_conv


@dataclass
class ConvHyperparams:
    """Conv hyperparameters."""

    strides: Union[Sequence[int], int] = 1
    padding: Union[str, Sequence[tuple[int, int]]] = "VALID"


def conv1d(lhs: Tensor, rhs: Tensor, config_obj: typing.Optional[Any] = None, **kwargs: Any) -> Any:  # type: ignore
    """1D Convolution.

    Args:
        lhs (Tensor): The lhs parameter.
        rhs (Tensor): The rhs parameter.
        config_obj (object): The config_obj parameter.
        **kwargs (object): Keyword args.

    Returns:
        Tensor: Result.
    """
    if config_obj is None:
        config_obj = _build_conv_config(kwargs, ((0, 2, 1), (2, 1, 0), (0, 2, 1)))

    return conv_general_dilated(lhs, rhs, config_obj)


def conv1d_transpose(
    lhs: Tensor,  # type: ignore
    rhs: Tensor,  # type: ignore
    strides: Union[Sequence[int], int] = 1,
    padding: Union[str, Sequence[tuple[int, int]]] = "VALID",
) -> Any:
    """1D convolution transpose.

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


def depthwise_conv1d(lhs: Tensor, rhs: Tensor, config_obj: typing.Optional[Any] = None, **kwargs: Any) -> Any:  # type: ignore
    """1D Depthwise Convolution.

    Args:
        lhs (Tensor): The lhs parameter.
        rhs (Tensor): The rhs parameter.
        config_obj (object): The config_obj parameter.
        **kwargs (object): Keyword args.

    Returns:
        Tensor: Result.
    """
    dimension_numbers = ((0, 2, 1), (2, 1, 0), (0, 2, 1))
    rhs_reshaped, config_obj = _prepare_depthwise_conv(lhs, rhs, 1, dimension_numbers, config_obj, **kwargs)
    return conv_general_dilated(lhs, rhs_reshaped, config_obj)


def separable_conv1d(
    lhs: Tensor,  # type: ignore
    depthwise_filter: Tensor,  # type: ignore
    pointwise_filter: Tensor,  # type: ignore
    config: Any = None,
    **kwargs: Any,
) -> Any:
    """1D Separable Convolution.

    Args:
        lhs (Tensor): The lhs parameter.
        depthwise_filter (Tensor): The depthwise_filter parameter.
        pointwise_filter (Tensor): The pointwise_filter parameter.
        config (ConvHyperparams): The config parameter.
        **kwargs (object): Keyword args.

    Returns:
        Tensor: Result.
    """
    config = config or ConvHyperparams()
    strides, padding = config.strides, config.padding
    kwargs["strides"] = strides
    kwargs["padding"] = padding
    depthwise_out = depthwise_conv1d(lhs, depthwise_filter, None, **kwargs)
    return conv1d(depthwise_out, pointwise_filter, None, strides=1, padding="VALID")
