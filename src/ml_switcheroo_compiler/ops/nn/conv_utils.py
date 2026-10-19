"""Convolution operations."""

from __future__ import annotations

import math
import typing
from collections.abc import Sequence
from dataclasses import dataclass
from typing import (
    Optional,
    Union,  # noqa: F401
    cast,
)

from ml_switcheroo_compiler.backends.registry import EagerCallableProtocol
from ml_switcheroo_compiler.core.shape import TensorOrScalar

# Base logic implementation
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.binary import add, multiply
from ml_switcheroo_compiler.ops.configs import ConvConfig
from ml_switcheroo_compiler.ops.reductions import mean
from ml_switcheroo_compiler.ops.registry import get_op
from ml_switcheroo_compiler.ops.shape.frontend import reshape

# Base logic implementation
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
    if s > k - 1:
        pad_a = k - 1
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
    pad_len = k + s - 2 + max(k - s, 0)
    pad_a = k - 1
    return pad_a, pad_len - pad_a


def _calculate_conv_transpose_padding(
    padding: Union[str, Sequence[tuple[int, int]]],
    k_sdims: tuple[int, ...],
    strides_tuple: tuple[int, ...],
) -> Union[str, Sequence[tuple[int, int]]]:
    """Calculate padding for transposed convolution.

    Args:
        padding (Union[str, Sequence[tuple[int, int]]]): Padding mode or sequence.
        k_sdims (tuple[int, ...]): Kernel spatial dimensions.
        strides_tuple (tuple[int, ...]): Strides tuple.

    Returns:
        Union[str, Sequence[tuple[int, int]]]: The calculated padding sequence.
    """
    if not (isinstance(padding, str) and padding in {"SAME", "VALID"}):
        return padding

    pads: list[tuple[int, int]] = []
    for k, s in zip(k_sdims, strides_tuple):
        if padding == "SAME":
            pads.append(_calc_same_pad(k, s))
        else:  # VALID
            pads.append(_calc_valid_pad(k, s))
    return pads


def _build_conv_config(
    kwargs: dict[str, int | str | Sequence[int] | Sequence[tuple[int, int]] | None],
    dimension_numbers: tuple[str, str, str] | tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
) -> ConvConfig:
    """Compute the build conv config tensor operation.

    Args:
        kwargs (dict): The kwargs parameter.
        dimension_numbers (tuple): The dimension_numbers parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    strides = kwargs.get("strides", 1)
    if isinstance(strides, int):
        strides = (strides,) * (len(dimension_numbers[0]) - 2)
    lhs_dilation = kwargs.get("lhs_dilation", None)
    if isinstance(lhs_dilation, int):
        lhs_dilation = (lhs_dilation,) * (len(dimension_numbers[0]) - 2)
    rhs_dilation = kwargs.get("rhs_dilation", None)
    if isinstance(rhs_dilation, int):
        rhs_dilation = (rhs_dilation,) * (len(dimension_numbers[0]) - 2)
    return cast(
        ConvConfig,
        cast("EagerCallableProtocol", ConvConfig)(
            window_strides=cast("Sequence[int]", strides),
            padding=kwargs.get("padding", "VALID"),
            lhs_dilation=cast("Optional[Sequence[int]]", lhs_dilation),
            rhs_dilation=cast("Optional[Sequence[int]]", rhs_dilation),
            dimension_numbers=dimension_numbers,
        ),
    )


def _prepare_depthwise_conv(
    lhs: Tensor,
    rhs: Tensor,
    spatial_dims: int,
    dimension_numbers: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    config_obj: TensorOrScalar = None,
    **kwargs: TensorOrScalar,
) -> TensorOrScalar:
    """Prepare configuration and reshape weights for depthwise convolution.

    Args:
        lhs (Tensor): The lhs parameter.
        rhs (Tensor): The rhs parameter.
        spatial_dims (int): The spatial_dims parameter.
        dimension_numbers (tuple): The dimension_numbers parameter.
        config_obj (TensorOrScalar): The config_obj parameter.
        **kwargs (TensorOrScalar): Keyword args.

    Returns:
        tuple: Result.
    """
    if config_obj is None:
        strides = kwargs.get("strides", 1)
        if isinstance(strides, int):
            strides = (strides,) * spatial_dims
        lhs_dilation = kwargs.get("lhs_dilation", None)
        if isinstance(lhs_dilation, int):
            lhs_dilation = (lhs_dilation,) * spatial_dims
        rhs_dilation = kwargs.get("rhs_dilation", None)
        if isinstance(rhs_dilation, int):
            rhs_dilation = (rhs_dilation,) * spatial_dims

        in_channels = lhs.shape[-1]

        config_obj = cast(
            TensorOrScalar,
            cast("EagerCallableProtocol", ConvConfig)(
                window_strides=cast("Sequence[int]", strides),
                padding=kwargs.get("padding", "VALID"),
                lhs_dilation=cast("Optional[Sequence[int]]", lhs_dilation),
                rhs_dilation=cast("Optional[Sequence[int]]", rhs_dilation),
                dimension_numbers=dimension_numbers,
                feature_group_count=in_channels,
            ),
        )

    in_channels = rhs.shape[-2]
    channel_multiplier = rhs.shape[-1]

    new_rhs_shape = rhs.shape[:spatial_dims] + (1, in_channels * channel_multiplier)
    rhs_reshaped = reshape(rhs, new_rhs_shape)

    return rhs_reshaped, config_obj


def atrous_conv2d(value: TensorOrScalar, filters: TensorOrScalar, rate: TensorOrScalar, padding: TensorOrScalar, name: TensorOrScalar = None) -> TensorOrScalar:
    """Atrous convolution.

    Args:
        value (TensorOrScalar): The value parameter.
        filters (TensorOrScalar): The filters parameter.
        rate (TensorOrScalar): The rate parameter.
        padding (TensorOrScalar): The padding parameter.
        name (TensorOrScalar): The name parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    conv2d = get_op("Conv2d")()

    return conv2d(value, filters, strides=1, padding=padding, dilation_rate=rate)


def atrous_conv2d_transpose(
    value: TensorOrScalar,
    filters: TensorOrScalar,
    output_shape: TensorOrScalar,
    config: typing.Optional[GenericConvConfig] = None,
    name: TensorOrScalar = None,
) -> TensorOrScalar:
    """Atrous convolution transpose.

    Args:
        value (TensorOrScalar): The value parameter.
        filters (TensorOrScalar): The filters parameter.
        output_shape (TensorOrScalar): The output_shape parameter.
        config (typing.Optional[GenericConvConfig]): The config parameter.
        name (TensorOrScalar): The name parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    conf = config if config is not None else GenericConvConfig()

    conv2d_transpose = get_op("Conv2dTranspose")()

    return conv2d_transpose(value, filters, strides=1, padding=conf.padding, dilation_rate=conf.dilation_rate)


def bias_add(value: TensorOrScalar, bias: TensorOrScalar, data_format: TensorOrScalar = None, name: TensorOrScalar = None) -> TensorOrScalar:
    """Add `bias` to `value`.

    Args:
        value (TensorOrScalar): The value parameter.
        bias (TensorOrScalar): The bias parameter.
        data_format (TensorOrScalar): The data_format parameter.
        name (TensorOrScalar): The name parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return add(value, bias)


def collapse_repeated(labels: TensorOrScalar, seq_length: TensorOrScalar, name: TensorOrScalar = None) -> TensorOrScalar:
    """Merge repeated labels into single labels.

    Args:
        labels (TensorOrScalar): The labels parameter.
        seq_length (TensorOrScalar): The seq_length parameter.
        name (TensorOrScalar): The name parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("CollapseRepeated", labels, seq_length=seq_length, name=name)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    attrs = {"seq_length": seq_length, "name": name}
    out = _emit_shape_node("CollapseRepeated", [labels], attrs, getattr(labels, "shape", ()), getattr(labels, "dtype", "int32"))
    new_seq_len = _emit_shape_node("CollapseRepeatedSeqLength", [labels], attrs, getattr(labels, "shape", ()), "int32")
    return out, new_seq_len


def compute_average_loss(per_example_loss: TensorOrScalar, sample_weight: TensorOrScalar = None, global_batch_size: TensorOrScalar = None) -> TensorOrScalar:
    """Compute the average loss.

    Args:
        per_example_loss (TensorOrScalar): The per_example_loss parameter.
        sample_weight (TensorOrScalar): The sample_weight parameter.
        global_batch_size (TensorOrScalar): The global_batch_size parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    if sample_weight is not None:
        per_example_loss = multiply(per_example_loss, sample_weight)
    return mean(per_example_loss)


def depthwise_conv2d(
    input: Tensor,
    filter: Tensor,
    config: typing.Optional[GenericConvConfig] = None,
    name: typing.Optional[str] = None,
) -> TensorOrScalar:
    """Depthwise 2-D convolution.

    Args:
        input (Tensor): The input parameter.
        filter (Tensor): The filter parameter.
        config (typing.Optional[GenericConvConfig]): The config parameter.
        name (typing.Optional[str]): The name parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    conf = config if config is not None else GenericConvConfig()

    conv2d = get_op("Conv2d")()

    return conv2d(
        input,
        filter,
        strides=conf.strides,
        padding=conf.padding,
        dilation_rate=conf.dilation_rate,
        groups=filter.shape[2],
    )


def depthwise_conv2d_backprop_filter(
    input: TensorOrScalar,
    filter_sizes: TensorOrScalar,
    out_backprop: TensorOrScalar,
    config: typing.Optional[GenericConvConfig] = None,
    name: TensorOrScalar = None,
) -> TensorOrScalar:
    """Compute the gradients of depthwise convolution with respect to the filter.

    Args:
        input (TensorOrScalar): Input activation tensor.
        filter_sizes (TensorOrScalar): Shape tuple of the filter tensor.
        out_backprop (TensorOrScalar): Upstream gradient tensor with respect to convolution output.
        config (Optional[GenericConvConfig]): Optional convolution configuration.
        name (TensorOrScalar): Optional operation name.

    Returns:
        TensorOrScalar: Filter gradient tensor of shape `filter_sizes`.
    """
    from ml_switcheroo_compiler.core.config import config as core_config

    raw_filter_shape = getattr(filter_sizes, "shape", None)
    target_shape: tuple[int | str, ...] = (int(filter_sizes),) if isinstance(filter_sizes, int) else tuple(cast("Sequence[Union[int, str]]", filter_sizes)) if isinstance(filter_sizes, (list, tuple)) else tuple(raw_filter_shape) if isinstance(raw_filter_shape, (list, tuple)) else ()
    dtype = getattr(input, "dtype", "float32")
    if core_config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("DepthwiseConv2dNativeBackpropFilter", getattr(input, "data", input), filter_sizes, getattr(out_backprop, "data", out_backprop), config=config, name=name)
        return Tensor(data, TensorConfig(target_shape, dtype, getattr(input, "device", "cpu")))
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node(
        "DepthwiseConv2dNativeBackpropFilter",
        [input, filter_sizes, out_backprop],
        {"config": config, "name": name},
        target_shape,
        dtype,
    )


def depthwise_conv2d_backprop_input(
    input_sizes: TensorOrScalar,
    filter: TensorOrScalar,
    out_backprop: TensorOrScalar,
    config: typing.Optional[GenericConvConfig] = None,
    name: TensorOrScalar = None,
) -> TensorOrScalar:
    """Compute the gradients of depthwise convolution with respect to the input.

    Args:
        input_sizes (TensorOrScalar): Shape tuple of the input tensor.
        filter (TensorOrScalar): Depthwise filter tensor.
        out_backprop (TensorOrScalar): Upstream gradient tensor with respect to convolution output.
        config (Optional[GenericConvConfig]): Optional convolution configuration.
        name (TensorOrScalar): Optional operation name.

    Returns:
        TensorOrScalar: Input gradient tensor of shape `input_sizes`.
    """
    from ml_switcheroo_compiler.core.config import config as core_config

    raw_input_shape = getattr(input_sizes, "shape", None)
    target_shape: tuple[int | str, ...] = (int(input_sizes),) if isinstance(input_sizes, int) else tuple(cast("Sequence[Union[int, str]]", input_sizes)) if isinstance(input_sizes, (list, tuple)) else tuple(raw_input_shape) if isinstance(raw_input_shape, (list, tuple)) else ()
    dtype = getattr(filter, "dtype", "float32")
    if core_config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("DepthwiseConv2dNativeBackpropInput", input_sizes, getattr(filter, "data", filter), getattr(out_backprop, "data", out_backprop), config=config, name=name)
        return Tensor(data, TensorConfig(target_shape, dtype, getattr(filter, "device", "cpu")))
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node(
        "DepthwiseConv2dNativeBackpropInput",
        [input_sizes, filter, out_backprop],
        {"config": config, "name": name},
        target_shape,
        dtype,
    )


@dataclass
class ConvSpatialArgs:
    """Arguments for spatial convolutions and morphological ops."""

    strides = None
    padding = "VALID"
    data_format = None
    dilations = None
    rates = None
    name = None


def dilation2d(
    input: TensorOrScalar,
    filter: TensorOrScalar,
    args: typing.Optional[ConvSpatialArgs] = None,
) -> TensorOrScalar:
    """Compute the grayscale dilation of 4-D `input` and 3-D `filter` tensors.

    Args:
        input (TensorOrScalar): The input parameter.
        filter (TensorOrScalar): The filter parameter.
        args (typing.Optional[ConvSpatialArgs]): The args parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Dilation2d", input, filter, args=args)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Dilation2d", [input, filter], {"args": args}, getattr(input, "shape", ()), getattr(input, "dtype", "float32"))


def erosion2d(
    value: TensorOrScalar,
    kernel: TensorOrScalar,
    args: typing.Optional[ConvSpatialArgs] = None,
) -> TensorOrScalar:
    """Compute the grayscale erosion of 4-D `value` and 3-D `kernel` tensors.

    Args:
        value (TensorOrScalar): The value parameter.
        kernel (TensorOrScalar): The kernel parameter.
        args (typing.Optional[ConvSpatialArgs]): The args parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Erosion2d", value, kernel, args=args)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Erosion2d", [value, kernel], {"args": args}, getattr(value, "shape", ()), getattr(value, "dtype", "float32"))


def convolution(
    input: TensorOrScalar,
    filters: TensorOrScalar,
    args: typing.Optional[ConvSpatialArgs] = None,
) -> TensorOrScalar:
    """Compute sums of N-D convolutions (actually cross-correlation).

    Args:
        input (TensorOrScalar): The input parameter.
        filters (TensorOrScalar): The filters parameter.
        args (typing.Optional[ConvSpatialArgs]): The args parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    args = args or ConvSpatialArgs()
    _conv_nd = get_op("_ConvNd")()

    num_spatial_dims = len(getattr(input, "shape")) - 2
    return _conv_nd(
        input,
        filters,
        strides=args.strides,
        padding=args.padding,
        dilation_rate=args.dilations,
        groups=1,
        num_spatial_dims=num_spatial_dims,
    )


def conv_transpose(
    input: TensorOrScalar,
    filters: TensorOrScalar,
    output_shape: TensorOrScalar,
    args: typing.Optional[ConvSpatialArgs] = None,
) -> TensorOrScalar:
    """Return the transpose of `convolution`.

    Args:
        input (TensorOrScalar): The input parameter.
        filters (TensorOrScalar): The filters parameter.
        output_shape (TensorOrScalar): The output_shape parameter.
        args (typing.Optional[ConvSpatialArgs]): The args parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("ConvTranspose", input, filters, output_shape=output_shape, args=args)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("ConvTranspose", [input, filters], {"output_shape": output_shape, "args": args}, cast("Sequence[Union[int, str]]", output_shape), getattr(input, "dtype", "float32"))
