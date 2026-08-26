# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Convolution operations."""

import typing
from collections.abc import Sequence
from typing import Union

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.linalg import conv_general_dilated
from ml_switcheroo_compiler.ops.registry import get_op

from .conv_utils import _build_conv_config


def conv3d(lhs: Tensor, rhs: Tensor, config_obj=None, **kwargs):
    """3D Convolution.

    Args:
        lhs (Tensor): Left-hand side tensor (batch, depth, height, width, in_channels).
        rhs (Tensor): Right-hand side tensor (depth, height, width, in_channels, out_channels).
        config_obj (ConvConfig | None): Configuration.
        **kwargs: Additional kwargs.

    Returns:
        Tensor: The result of the convolution.
    """
    if config_obj is None:
        config_obj = _build_conv_config(kwargs, ((0, 4, 1, 2, 3), (4, 3, 0, 1, 2), (0, 4, 1, 2, 3)))

    return conv_general_dilated(lhs, rhs, config_obj)


def conv3d_transpose(
    lhs: Tensor,
    rhs: Tensor,
    strides: Union[Sequence[int], int] = 1,
    padding: Union[str, Sequence[tuple[int, int]]] = "VALID",
):
    """3D convolution transpose.

    Args:
        lhs (Tensor): Left-hand side tensor.
        rhs (Tensor): Right-hand side tensor.
        strides (Union[Sequence[int], int]): Strides.
        padding (Union[str, Sequence[tuple[int, int]]]): Padding.

    Returns:
        Tensor: The result of the convolution.
    """
    conv_transpose = get_op("ConvTranspose")()

    return conv_transpose(lhs, rhs, strides, padding)
