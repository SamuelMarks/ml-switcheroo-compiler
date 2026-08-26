# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Convolution operations."""

import typing
from collections.abc import Sequence
from typing import Union

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.constants import MAGIC_VAL_2, MAGIC_VAL_3
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.linalg.utils import _emit_linalg_node

from .conv1d import conv1d, depthwise_conv1d, separable_conv1d
from .conv2d import conv2d, depthwise_conv2d, separable_conv2d
from .conv3d import conv3d
from .conv_utils import GenericConvConfig


def conv_transpose(
    lhs: Tensor,
    rhs: Tensor,
    strides: Union[Sequence[int], int] = 1,
    padding: Union[str, Sequence[tuple[int, int]]] = "VALID",
):
    """Convolution transpose.

    Args:
        lhs (Tensor): Left-hand side tensor.
        rhs (Tensor): Right-hand side tensor.
        strides (Union[Sequence[int], int]): Strides.
        padding (Union[str, Sequence[tuple[int, int]]]): Padding.

    Returns:
        Tensor: The result of the convolution.
    """
    op_name = "ConvTranspose"

    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op(
            op_name,
            getattr(lhs, "data", lhs),
            getattr(rhs, "data", rhs),
            strides=strides,
            padding=padding,
        )
        return Tensor(data, TensorConfig(getattr(data, "shape", ()), lhs.dtype, lhs.device))

    return _emit_linalg_node(op_name, [lhs, rhs], {"strides": strides, "padding": padding}, [()], [lhs.dtype])


def conv(
    inputs: Tensor,
    kernel: Tensor,
    config: typing.Optional[GenericConvConfig] = None,
):
    """Evaluate conv operation.

    Args:
        inputs (Tensor): The inputs parameter.
        kernel (Tensor): The kernel parameter.
        config (object): The config parameter.

    Returns:
        Tensor: Result.

    Raises:
        ValueError: An exception.
    """
    conf = config if config is not None else GenericConvConfig()
    """Provide generic convolution.

    Args:
        inputs (Tensor): The input tensor.
        kernel (Tensor): The kernel tensor.
        strides (Union[int, Sequence[int]]): Strides.
        padding (Union[str, Sequence[tuple[int, int]]]): Padding.
        data_format (Optional[str]): Data format.
        dilation_rate (Union[int, Sequence[int]]): Dilation rate.

    Returns:
        Tensor: Convolution output.
    """
    spatial_rank = len(inputs.shape) - 2
    if spatial_rank == 1:
        return conv1d(
            inputs,
            kernel,
            strides=conf.strides,
            padding=conf.padding,
            lhs_dilation=1,
            rhs_dilation=conf.dilation_rate,
        )
    elif spatial_rank == MAGIC_VAL_2:
        return conv2d(
            inputs,
            kernel,
            strides=conf.strides,
            padding=conf.padding,
            lhs_dilation=1,
            rhs_dilation=conf.dilation_rate,
        )
    elif spatial_rank == MAGIC_VAL_3:
        return conv3d(
            inputs,
            kernel,
            strides=conf.strides,
            padding=conf.padding,
            lhs_dilation=1,
            rhs_dilation=conf.dilation_rate,
        )
    else:
        raise ValueError(f"Unsupported spatial rank: {spatial_rank}")


def depthwise_conv(
    inputs: Tensor,
    kernel: Tensor,
    config: typing.Optional[GenericConvConfig] = None,
):
    """Evaluate depthwise_conv operation.

    Args:
        inputs (Tensor): The inputs parameter.
        kernel (Tensor): The kernel parameter.
        config (object): The config parameter.

    Returns:
        Tensor: Result.

    Raises:
        ValueError: An exception.
    """
    conf = config if config is not None else GenericConvConfig()
    """Provide generic depthwise convolution.

    Args:
        inputs (Tensor): Input tensor.
        kernel (Tensor): Kernel tensor.
        strides (Union[int, Sequence[int]]): Strides.
        padding (Union[str, Sequence[tuple[int, int]]]): Padding.
        data_format (Optional[str]): Data format.
        dilation_rate (Union[int, Sequence[int]]): Dilation rate.

    Returns:
        Tensor: Convolution output.
    """
    spatial_rank = len(inputs.shape) - 2
    if spatial_rank == 1:
        return depthwise_conv1d(
            inputs,
            kernel,
            strides=conf.strides,
            padding=conf.padding,
            lhs_dilation=1,
            rhs_dilation=conf.dilation_rate,
        )
    elif spatial_rank == MAGIC_VAL_2:
        return depthwise_conv2d(
            inputs,
            kernel,
            config=GenericConvConfig(
                strides=conf.strides,
                padding=conf.padding,
                dilation_rate=conf.dilation_rate,
            ),
        )
    else:
        raise ValueError(f"Unsupported spatial rank for depthwise_conv: {spatial_rank}")


def separable_conv(
    inputs: Tensor,
    depthwise_kernel: Tensor,
    pointwise_kernel: Tensor,
    config: typing.Optional[GenericConvConfig] = None,
):
    """Evaluate separable_conv operation.

    Args:
        inputs (Tensor): The inputs parameter.
        depthwise_kernel (Tensor): The depthwise_kernel parameter.
        pointwise_kernel (Tensor): The pointwise_kernel parameter.
        config (object): The config parameter.

    Returns:
        Tensor: Result.

    Raises:
        ValueError: An exception.
    """
    conf = config if config is not None else GenericConvConfig()
    """Provide generic separable convolution.

    Args:
        inputs (Tensor): Input tensor.
        depthwise_kernel (Tensor): Depthwise kernel.
        pointwise_kernel (Tensor): Pointwise kernel.
        strides (Union[int, Sequence[int]]): Strides.
        padding (Union[str, Sequence[tuple[int, int]]]): Padding.
        data_format (Optional[str]): Data format.
        dilation_rate (Union[int, Sequence[int]]): Dilation rate.

    Returns:
        Tensor: Convolution output.
    """
    spatial_rank = len(inputs.shape) - 2
    if spatial_rank == 1:
        return separable_conv1d(
            inputs,
            depthwise_kernel,
            pointwise_kernel,
            strides=conf.strides,
            padding=conf.padding,
            lhs_dilation=1,
            rhs_dilation=conf.dilation_rate,
        )
    elif spatial_rank == MAGIC_VAL_2:
        return separable_conv2d(
            inputs,
            depthwise_kernel,
            pointwise_kernel,
            strides=conf.strides,
            padding=conf.padding,
            lhs_dilation=1,
            rhs_dilation=conf.dilation_rate,
        )
    else:
        raise ValueError(f"Unsupported spatial rank for separable_conv: {spatial_rank}")
