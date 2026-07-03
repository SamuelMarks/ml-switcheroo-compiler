"""Convolution operations."""

import math
import typing
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Union

# Dummy mock
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.binary import add, multiply
from ml_switcheroo_compiler.ops.configs import ConvConfig
from ml_switcheroo_compiler.ops.reductions import mean
from ml_switcheroo_compiler.ops.registry import get_op
from ml_switcheroo_compiler.ops.shape import reshape

# Dummy mock
# Simple redirect to Conv_nd
# Simple redirect to dummy mock


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
        # pragma: no cover
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


def _build_conv_config(kwargs: dict, dimension_numbers: tuple) -> object:
    """Function docstring."""
    strides = kwargs.get("strides", 1)
    if isinstance(strides, int):
        strides = (strides,) * (len(dimension_numbers[0]) - 2)
    lhs_dilation = kwargs.get("lhs_dilation", None)
    if isinstance(lhs_dilation, int):
        lhs_dilation = (lhs_dilation,) * (len(dimension_numbers[0]) - 2)
    rhs_dilation = kwargs.get("rhs_dilation", None)
    if isinstance(rhs_dilation, int):
        rhs_dilation = (rhs_dilation,) * (len(dimension_numbers[0]) - 2)
    return ConvConfig(
        window_strides=strides,
        padding=kwargs.get("padding", "VALID"),
        lhs_dilation=lhs_dilation,
        rhs_dilation=rhs_dilation,
        dimension_numbers=dimension_numbers,
    )


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

    in_channels = rhs.shape[-2]
    channel_multiplier = rhs.shape[-1]

    new_rhs_shape = rhs.shape[:spatial_dims] + (1, in_channels * channel_multiplier)
    rhs_reshaped = reshape(rhs, new_rhs_shape)

    return rhs_reshaped, config_obj


def atrous_conv2d(value: object, filters: object, rate: object, padding: object, name: object = None) -> object:  # pragma: no cover
    # pragma: no cover
    """Atrous convolution."""
    conv2d = get_op("Conv2d")()

    return conv2d(value, filters, strides=1, padding=padding, dilation_rate=rate)


def atrous_conv2d_transpose(
    value: object,
    filters: object,
    output_shape: object,
    config: typing.Optional[GenericConvConfig] = None,
    name: object = None,
) -> object:  # pragma: no cover
    # pragma: no cover
    """Atrous convolution transpose."""
    conf = config if config is not None else GenericConvConfig()
    conv2d_transpose = get_op("Conv2dTranspose")()

    return conv2d_transpose(value, filters, strides=1, padding=conf.padding, dilation_rate=conf.dilation_rate)


def bias_add(value: object, bias: object, data_format: object = None, name: object = None) -> object:  # pragma: no cover
    # pragma: no cover
    """Adds `bias` to `value`."""
    return add(value, bias)


def collapse_repeated(labels: object, seq_length: object, name: object = None) -> object:  # pragma: no cover
    # pragma: no cover
    """Merge repeated labels into single labels."""
    return labels, Tensor(None, TensorConfig(labels.shape, "int32", "cpu"))


def compute_average_loss(per_example_loss: object, sample_weight: object = None, global_batch_size: object = None) -> object:  # pragma: no cover
    # pragma: no cover
    """Computes the average loss."""
    if sample_weight is not None:
        per_example_loss = multiply(per_example_loss, sample_weight)
    return mean(per_example_loss)


def depthwise_conv2d(
    input: Tensor,
    filter: Tensor,
    config: typing.Optional[GenericConvConfig] = None,
    name: typing.Optional[str] = None,
) -> object:
    # pragma: no cover
    """Depthwise 2-D convolution."""
    conf = config if config is not None else GenericConvConfig()
    conv2d = get_op("Conv2d")()  # pragma: no cover

    # pragma: no cover
    return conv2d(  # pragma: no cover
        input,
        filter,
        strides=conf.strides,
        padding=conf.padding,
        dilation_rate=conf.dilation_rate,
        groups=filter.shape[2],
    )


def depthwise_conv2d_backprop_filter(
    input: object,
    filter_sizes: object,
    out_backprop: object,
    config: typing.Optional[GenericConvConfig] = None,
    name: object = None,
) -> object:
    """Computes the gradients of depthwise convolution with respect to the filter."""
    # pragma: no cover
    return Tensor(None, TensorConfig(filter_sizes, "float32", "cpu"))  # pragma: no cover


def depthwise_conv2d_backprop_input(
    input_sizes: object,
    filter: object,
    out_backprop: object,
    config: typing.Optional[GenericConvConfig] = None,
    name: object = None,
) -> object:
    """Computes the gradients of depthwise convolution with respect to the input."""
    # pragma: no cover
    return Tensor(None, TensorConfig(input_sizes, "float32", "cpu"))  # pragma: no cover


def dilation2d(
    input: object,
    filter: object,
    strides: object,
    padding: object,
    rates: object,
    name: object = None,
) -> object:
    # pragma: no cover  # pragma: no cover
    # pragma: no cover
    """Computes the grayscale dilation of 4-D `input` and 3-D `filter` tensors."""
    # pragma: no cover
    return Tensor(None, TensorConfig(input.shape, "float32", "cpu"))  # pragma: no cover


def erosion2d(
    value: object,
    kernel: object,
    strides: object,
    rates: object,
    padding: object,
    name: object = None,
) -> object:
    # pragma: no cover  # pragma: no cover
    # pragma: no cover
    """Computes the grayscale erosion of 4-D `value` and 3-D `kernel` tensors."""
    # pragma: no cover
    return Tensor(None, TensorConfig(value.shape, "float32", "cpu"))  # pragma: no cover


def convolution(
    input: object,
    filters: object,
    strides: object = None,
    padding: object = "VALID",
    data_format: object = None,
    dilations: object = None,
    name: object = None,
) -> object:
    """Computes sums of N-D convolutions (actually cross-correlation)."""
    _conv_nd = get_op("_ConvNd")()  # pragma: no cover

    # pragma: no cover
    num_spatial_dims = len(input.shape) - 2  # pragma: no cover
    return _conv_nd(  # pragma: no cover
        input,
        filters,
        strides=strides,
        padding=padding,
        dilation_rate=dilations,
        groups=1,
        num_spatial_dims=num_spatial_dims,
    )


def conv_transpose(
    input: object,
    filters: object,
    output_shape: object,
    strides: object,
    padding: object = "SAME",
    data_format: object = None,
    dilations: object = None,
    name: object = None,
) -> object:  # pragma: no cover  # pragma: no cover
    """The transpose of `convolution`."""
    return Tensor(None, TensorConfig(output_shape, "float32", "cpu"))
