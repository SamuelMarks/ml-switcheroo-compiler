"""Convolution operations."""

import typing
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.linalg import conv_general_dilated
from collections.abc import Sequence
from typing import Union

from .conv_utils import _prepare_depthwise_conv


def conv2d(
    lhs: Tensor, rhs: Tensor, config_obj: typing.Optional[object] = None, **kwargs: object
) -> Tensor:
    """2D Convolution.

    Args:
        lhs (Tensor): Left-hand side tensor (batch, height, width, in_channels).
        rhs (Tensor): Right-hand side tensor (height, width, in_channels, out_channels).
        config_obj (ConvConfig | None): Configuration.
        **kwargs: Backward compatibility arguments.

    Returns:
        Tensor: The result of the convolution.
    """
    if config_obj is None:  # pragma: no branch
        from ml_switcheroo_compiler.ops.configs import ConvConfig

        strides = kwargs.get("strides", 1)
        if isinstance(strides, int):
            strides = (strides, strides)
        lhs_dilation = kwargs.get("lhs_dilation", None)
        if isinstance(lhs_dilation, int):
            lhs_dilation = (lhs_dilation, lhs_dilation)
        rhs_dilation = kwargs.get("rhs_dilation", None)
        if isinstance(rhs_dilation, int):
            rhs_dilation = (rhs_dilation, rhs_dilation)
        config_obj = ConvConfig(
            window_strides=strides,
            padding=kwargs.get("padding", "VALID"),
            lhs_dilation=lhs_dilation,
            rhs_dilation=rhs_dilation,
            dimension_numbers=((0, 3, 1, 2), (3, 2, 0, 1), (0, 3, 1, 2)),
        )

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

    Returns:
        Tensor: The result of the convolution.
    """
    from .conv_nd import conv_transpose

    return conv_transpose(lhs, rhs, strides, padding)


def depthwise_conv2d(
    lhs: Tensor, rhs: Tensor, config_obj: typing.Optional[object] = None, **kwargs: object
) -> Tensor:
    """2D Depthwise Convolution.

    Args:
        lhs (Tensor): Left-hand side tensor (batch, height, width, in_channels).
        rhs (Tensor): Right-hand side tensor (height, width, in_channels, channel_multiplier).
        config_obj (ConvConfig | None): Configuration.
        **kwargs: Backward compatibility arguments.

    Returns:
        Tensor: The result of the convolution.
    """
    dimension_numbers = ((0, 3, 1, 2), (3, 2, 0, 1), (0, 3, 1, 2))
    rhs_reshaped, config_obj = _prepare_depthwise_conv(
        lhs, rhs, 2, dimension_numbers, config_obj, **kwargs
    )
    return conv_general_dilated(lhs, rhs_reshaped, config_obj)


def separable_conv2d(
    lhs: Tensor,
    depthwise_filter: Tensor,
    pointwise_filter: Tensor,
    strides: Union[Sequence[int], int] = 1,
    padding: Union[str, Sequence[tuple[int, int]]] = "VALID",
    **kwargs: object,
) -> Tensor:
    """2D Separable Convolution.

    Args:
        lhs (Tensor): Left-hand side tensor (batch, height, width, in_channels).
        depthwise_filter (Tensor): Depthwise filter tensor.
        pointwise_filter (Tensor): Pointwise filter tensor.
        strides (Union[Sequence[int], int]): Strides.
        padding (Union[str, Sequence[tuple[int, int]]]): Padding.
        **kwargs: Additional kwargs.

    Returns:
        Tensor: The result of the separable convolution.
    """
    dw_out = depthwise_conv2d(lhs, depthwise_filter, strides=strides, padding=padding, **kwargs)
    return conv2d(dw_out, pointwise_filter, strides=1, padding="VALID")
