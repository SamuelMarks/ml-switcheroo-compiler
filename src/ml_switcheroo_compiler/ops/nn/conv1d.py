"""Convolution operations."""

import typing  # noqa: F401
from collections.abc import Sequence
from dataclasses import dataclass
from typing import (
    Union,  # noqa: F401
    cast,
)

from ml_switcheroo_compiler.backends.registry import EagerCallableProtocol
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.linalg import conv_general_dilated
from ml_switcheroo_compiler.ops.registry import get_op

from .conv_utils import _build_conv_config, _prepare_depthwise_conv


@dataclass
class ConvHyperparams:
    """Conv hyperparameters."""

    strides: Union[Sequence[int], int] = 1
    padding: Union[str, Sequence[tuple[int, int]]] = "VALID"


def conv1d(lhs: Tensor, rhs: Tensor, config_obj: TensorOrScalar = None, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """1D Convolution.

    Args:
        lhs (Tensor): The lhs parameter.
        rhs (Tensor): The rhs parameter.
        config_obj (TensorOrScalar): The config_obj parameter.
        **kwargs (TensorOrScalar): Keyword args.

    Returns:
        Tensor: Result.
    """
    if config_obj is None:
        config_obj = cast(TensorOrScalar, cast("EagerCallableProtocol", _build_conv_config)(kwargs, ((0, 2, 1), (2, 1, 0), (0, 2, 1))))

    return conv_general_dilated(lhs, rhs, config_obj)


def conv1d_transpose(
    lhs: Tensor,
    rhs: Tensor,
    strides: Union[Sequence[int], int] = 1,
    padding: Union[str, Sequence[tuple[int, int]]] = "VALID",
) -> TensorOrScalar:
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


def depthwise_conv1d(lhs: Tensor, rhs: Tensor, config_obj: TensorOrScalar = None, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """1D Depthwise Convolution.

    Args:
        lhs (Tensor): The lhs parameter.
        rhs (Tensor): The rhs parameter.
        config_obj (TensorOrScalar): The config_obj parameter.
        **kwargs (TensorOrScalar): Keyword args.

    Returns:
        Tensor: Result.
    """
    dimension_numbers = ((0, 2, 1), (2, 1, 0), (0, 2, 1))
    rhs_reshaped, config_obj = getattr(_prepare_depthwise_conv(lhs, rhs, 1, dimension_numbers, config_obj, **kwargs), "__iter__")()
    return conv_general_dilated(lhs, rhs_reshaped, config_obj)


def separable_conv1d(
    lhs: Tensor,
    depthwise_filter: Tensor,
    pointwise_filter: Tensor,
    config: TensorOrScalar = None,
    **kwargs: TensorOrScalar,
) -> TensorOrScalar:
    """1D Separable Convolution.

    Args:
        lhs (Tensor): The lhs parameter.
        depthwise_filter (Tensor): The depthwise_filter parameter.
        pointwise_filter (Tensor): The pointwise_filter parameter.
        config (ConvHyperparams): The config parameter.
        **kwargs (TensorOrScalar): Keyword args.

    Returns:
        Tensor: Result.
    """
    config = cast(TensorOrScalar, config or ConvHyperparams())
    strides, padding = getattr(config, "strides"), getattr(config, "padding")
    kwargs["strides"] = strides
    kwargs["padding"] = padding
    depthwise_out = depthwise_conv1d(lhs, depthwise_filter, None, **kwargs)
    return conv1d(cast("Tensor", depthwise_out), pointwise_filter, None, strides=1, padding="VALID")
