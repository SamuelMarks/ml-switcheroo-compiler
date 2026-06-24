"""Convolution operations."""

from ml_switcheroo_compiler.core.constants import MAGIC_VAL_2
from ml_switcheroo_compiler.core.constants import MAGIC_VAL_3

import typing
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.linalg import conv_general_dilated
from collections.abc import Sequence
from typing import Union

from .conv_utils import GenericConvConfig, _calculate_conv_transpose_padding
from .conv1d import conv1d, depthwise_conv1d, separable_conv1d
from .conv2d import conv2d, depthwise_conv2d, separable_conv2d
from .conv3d import conv3d


def conv_transpose(
    lhs: Tensor,
    rhs: Tensor,
    strides: Union[Sequence[int], int] = 1,
    padding: Union[str, Sequence[tuple[int, int]]] = "VALID",
) -> Tensor:
    """Convolution transpose.

    Args:
        lhs (Tensor): Left-hand side tensor.
        rhs (Tensor): Right-hand side tensor.
        strides (Union[Sequence[int], int]): Strides.
        padding (Union[str, Sequence[tuple[int, int]]]): Padding.

    Returns:
        Tensor: The result of the convolution.
    """
    spatial_dims = len(lhs.shape) - 2
    if isinstance(strides, int):
        strides_tuple = (strides,) * spatial_dims
    else:
        strides_tuple = tuple(strides)

    # For default layout, k_sdims is at the end. Assuming OIW / OIHW / OIDHW
    k_sdims = rhs.shape[2:]

    pads = _calculate_conv_transpose_padding(padding, k_sdims, strides_tuple)

    from ml_switcheroo_compiler.ops.configs import ConvConfig

    config_obj = ConvConfig(
        window_strides=(1,) * spatial_dims,
        padding=pads,
        lhs_dilation=strides_tuple,
    )

    return conv_general_dilated(lhs, rhs, config_obj)


def conv(
    inputs: Tensor,
    kernel: Tensor,
    config: typing.Optional[GenericConvConfig] = None,
) -> Tensor:
    """Docstring."""
    conf = config if config is not None else GenericConvConfig()
    """Generic convolution.

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
    elif spatial_rank == MAGIC_VAL_3:  # pragma: no branch
        return conv3d(
            inputs,
            kernel,
            strides=conf.strides,
            padding=conf.padding,
            lhs_dilation=1,
            rhs_dilation=conf.dilation_rate,
        )
    else:
        raise ValueError(f"Unsupported spatial rank: {spatial_rank}")  # pragma: no cover


def depthwise_conv(
    inputs: Tensor,
    kernel: Tensor,
    config: typing.Optional[GenericConvConfig] = None,
) -> Tensor:
    """Docstring."""
    conf = config if config is not None else GenericConvConfig()
    """Generic depthwise convolution.

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
    elif spatial_rank == MAGIC_VAL_2:  # pragma: no branch
        return depthwise_conv2d(
            inputs,
            kernel,
            strides=conf.strides,
            padding=conf.padding,
            lhs_dilation=1,
            rhs_dilation=conf.dilation_rate,
        )
    else:
        raise ValueError(
            f"Unsupported spatial rank for depthwise_conv: {spatial_rank}"
        )  # pragma: no cover


def separable_conv(
    inputs: Tensor,
    depthwise_kernel: Tensor,
    pointwise_kernel: Tensor,
    config: typing.Optional[GenericConvConfig] = None,
) -> Tensor:
    """Docstring."""
    conf = config if config is not None else GenericConvConfig()
    """Generic separable convolution.

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
    elif spatial_rank == MAGIC_VAL_2:  # pragma: no branch
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
        raise ValueError(
            f"Unsupported spatial rank for separable_conv: {spatial_rank}"
        )  # pragma: no cover
