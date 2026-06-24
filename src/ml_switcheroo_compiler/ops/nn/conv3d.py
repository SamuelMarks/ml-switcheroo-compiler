"""Convolution operations."""

import typing
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.linalg import conv_general_dilated
from collections.abc import Sequence
from typing import Union


def conv3d(
    lhs: Tensor, rhs: Tensor, config_obj: typing.Optional[object] = None, **kwargs: object
) -> Tensor:
    """3D Convolution.

    Args:
        lhs (Tensor): Left-hand side tensor (batch, depth, height, width, in_channels).
        rhs (Tensor): Right-hand side tensor (depth, height, width, in_channels, out_channels).
        config_obj (ConvConfig | None): Configuration.
        **kwargs: Backward compatibility arguments.

    Returns:
        Tensor: The result of the convolution.
    """
    if config_obj is None:  # pragma: no branch
        from ml_switcheroo_compiler.ops.configs import ConvConfig

        strides = kwargs.get("strides", 1)
        if isinstance(strides, int):
            strides = (strides, strides, strides)
        lhs_dilation = kwargs.get("lhs_dilation", None)
        if isinstance(lhs_dilation, int):
            lhs_dilation = (lhs_dilation, lhs_dilation, lhs_dilation)
        rhs_dilation = kwargs.get("rhs_dilation", None)
        if isinstance(rhs_dilation, int):
            rhs_dilation = (rhs_dilation, rhs_dilation, rhs_dilation)
        config_obj = ConvConfig(
            window_strides=strides,
            padding=kwargs.get("padding", "VALID"),
            lhs_dilation=lhs_dilation,
            rhs_dilation=rhs_dilation,
            dimension_numbers=((0, 4, 1, 2, 3), (4, 3, 0, 1, 2), (0, 4, 1, 2, 3)),
        )

    return conv_general_dilated(lhs, rhs, config_obj)


def conv3d_transpose(
    lhs: Tensor,
    rhs: Tensor,
    strides: Union[Sequence[int], int] = 1,
    padding: Union[str, Sequence[tuple[int, int]]] = "VALID",
) -> Tensor:
    """3D convolution transpose.

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
