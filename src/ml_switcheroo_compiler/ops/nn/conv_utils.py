"""Convolution operations."""

import math
import typing
from ml_switcheroo_compiler.core.tensor import Tensor
from collections.abc import Sequence
from typing import Union


from dataclasses import dataclass


@dataclass
class GenericConvConfig:
    """Conv config."""

    strides: Union[int, Sequence[int]] = 1
    padding: Union[str, Sequence[tuple[int, int]]] = "VALID"
    data_format: typing.Optional[str] = None
    dilation_rate: Union[int, Sequence[int]] = 1


def _calc_same_pad(k: int, s: int) -> tuple[int, int]:
    """Calc SAME pad.

    Args:
        k (int): Kernel.
        s (int): Stride.

    Returns:
        tuple[int, int]: Pads.
    """
    pad_len = k + s - 2
    if s > k - 1:  # pragma: no branch
        pad_a = k - 1  # pragma: no cover
    else:
        pad_a = int(math.ceil(pad_len / 2.0))
    return pad_a, pad_len - pad_a


def _calc_valid_pad(k: int, s: int) -> tuple[int, int]:
    """Calc VALID pad.

    Args:
        k (int): Kernel.
        s (int): Stride.

    Returns:
        tuple[int, int]: Pads.
    """
    pad_len = k + s - 2 + max(k - s, 0)  # pragma: no cover
    pad_a = k - 1  # pragma: no cover
    return pad_a, pad_len - pad_a  # pragma: no cover


def _calculate_conv_transpose_padding(
    padding: Union[str, Sequence[tuple[int, int]]],
    k_sdims: tuple[int, ...],
    strides_tuple: tuple[int, ...],
) -> Sequence[tuple[int, int]]:
    """Calculate padding for transposed convolution.

    Args:
        padding (Union[str, Sequence[tuple[int, int]]]): Padding mode or sequence.
        k_sdims (tuple[int, ...]): Kernel spatial dimensions.
        strides_tuple (tuple[int, ...]): Strides tuple.

    Returns:
        Sequence[tuple[int, int]]: The calculated padding sequence.
    """
    if not (isinstance(padding, str) and padding in {"SAME", "VALID"}):  # pragma: no branch
        return padding  # pragma: no cover

    pads = []
    for k, s in zip(k_sdims, strides_tuple):
        if padding == "SAME":  # pragma: no branch
            pads.append(_calc_same_pad(k, s))
        else:  # VALID
            pads.append(_calc_valid_pad(k, s))  # pragma: no cover
    return pads


def _prepare_depthwise_conv(
    lhs: Tensor,
    rhs: Tensor,
    spatial_dims: int,
    dimension_numbers: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    config_obj: typing.Optional[object] = None,
    **kwargs: object,
) -> tuple[Tensor, object]:
    """Prepare configuration and reshape weights for depthwise convolution."""
    if config_obj is None:  # pragma: no branch
        from ml_switcheroo_compiler.ops.configs import ConvConfig

        strides = kwargs.get("strides", 1)
        if isinstance(strides, int):  # pragma: no branch
            strides = (strides,) * spatial_dims
        lhs_dilation = kwargs.get("lhs_dilation", None)
        if isinstance(lhs_dilation, int):
            lhs_dilation = (lhs_dilation,) * spatial_dims
        rhs_dilation = kwargs.get("rhs_dilation", None)
        if isinstance(rhs_dilation, int):
            rhs_dilation = (rhs_dilation,) * spatial_dims

        in_channels = lhs.shape[-1]

        config_obj = ConvConfig(
            window_strides=strides,
            padding=kwargs.get("padding", "VALID"),
            lhs_dilation=lhs_dilation,
            rhs_dilation=rhs_dilation,
            dimension_numbers=dimension_numbers,
            feature_group_count=in_channels,
        )

    from ml_switcheroo_compiler.ops.shape import reshape

    in_channels = rhs.shape[-2]
    channel_multiplier = rhs.shape[-1]

    new_rhs_shape = rhs.shape[:spatial_dims] + (1, in_channels * channel_multiplier)
    rhs_reshaped = reshape(rhs, new_rhs_shape)

    return rhs_reshaped, config_obj
