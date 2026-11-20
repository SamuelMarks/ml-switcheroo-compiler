"""Convolution operations."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import pydantic

from ml_switcheroo_compiler.backends.registry import coerce_int, coerce_operand_list, coerce_opt_int_seq, coerce_shape
from ml_switcheroo_compiler.core.shape import (
    OpValue,
    ShapeProtocol,
    extract_data,
    extract_device,
    extract_dtype,
    extract_shape,
)

# Base logic implementation
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.binary import add, multiply
from ml_switcheroo_compiler.ops.configs import ConvConfig
from ml_switcheroo_compiler.ops.reductions import mean
from ml_switcheroo_compiler.ops.registry import get_op
from ml_switcheroo_compiler.ops.shape.api import reshape


def _normalize_conv_padding(pad: OpValue) -> str | Sequence[tuple[int, int]]:
    """Normalize convolution padding argument to string or `(low, high)` tuple sequence.

    Args:
        pad (OpValue): Raw padding argument.

    Returns:
        str | Sequence[tuple[int, int]]: Normalized padding specification.
    """
    return (
        pad
        if isinstance(pad, str)
        else tuple(
            (coerce_int(p[0]), coerce_int(p[1])) for p in coerce_operand_list(pad) if isinstance(p, (tuple, list))
        )
    )


# Base logic implementation
# Simple redirect to Conv_nd
# Simple redirect to dummy mock


class GenericConvConfig(pydantic.BaseModel):
    """Conv config."""

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    strides: int | Sequence[int] = 1
    padding: str | Sequence[tuple[int, int]] = "VALID"
    data_format: str | None = None
    dilation_rate: int | Sequence[int] = 1


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
    padding: str | Sequence[tuple[int, int]] | tuple[int, int],
    k_sdims: tuple[int, ...],
    strides_tuple: tuple[int, ...],
) -> str | Sequence[tuple[int, int]] | tuple[int, int]:
    """Calculate padding for transposed convolution.

    Args:
        padding (str | Sequence[tuple[int, int]] | tuple[int, int]): Padding mode or sequence.
        k_sdims (tuple[int, ...]): Kernel spatial dimensions.
        strides_tuple (tuple[int, ...]): Strides tuple.

    Returns:
        str | Sequence[tuple[int, int]] | tuple[int, int]: The calculated padding sequence.
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
    kwargs: Mapping[str, OpValue],
    dimension_numbers: tuple[str, str, str] | tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
) -> ConvConfig:
    """Compute the build conv config tensor operation.

    Args:
        kwargs (Mapping[str, OpValue]): Input `kwargs` argument (kwargs).
        dimension_numbers (tuple[str, str, str] | tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]):
            Input `dimension_numbers` argument (dimension numbers).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
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
    return ConvConfig(
        window_strides=coerce_shape(strides),
        padding=_normalize_conv_padding(kwargs.get("padding", "VALID")),
        lhs_dilation=coerce_opt_int_seq(lhs_dilation),
        rhs_dilation=coerce_opt_int_seq(rhs_dilation),
        dimension_numbers=dimension_numbers,
    )


def _prepare_depthwise_conv(
    lhs: Tensor,
    rhs: Tensor,
    spatial_dims: int,
    dimension_numbers: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    config_obj: ConvConfig | OpValue = None,
    **kwargs: OpValue,
) -> tuple[OpValue, ConvConfig | OpValue]:
    """Prepare configuration and reshape weights for depthwise convolution.

    Args:
        lhs (Tensor): Input `lhs` argument (lhs).
        rhs (Tensor): Input `rhs` argument (rhs).
        spatial_dims (int): Input `spatial_dims` argument (spatial dims).
        dimension_numbers (tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]):
            Input `dimension_numbers` argument (dimension numbers).
        config_obj (ConvConfig | OpValue): Input `config_obj` argument (config obj).
        **kwargs (OpValue): Keyword args.

    Returns:
        tuple: Computed `tuple` output.
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

        in_channels = extract_shape(lhs)[-1]

        config_obj = ConvConfig(
            window_strides=coerce_shape(strides),
            padding=_normalize_conv_padding(kwargs.get("padding", "VALID")),
            lhs_dilation=coerce_opt_int_seq(lhs_dilation),
            rhs_dilation=coerce_opt_int_seq(rhs_dilation),
            dimension_numbers=dimension_numbers,
            feature_group_count=in_channels,
        )

    rhs_shape = extract_shape(rhs)
    in_channels = rhs_shape[-2]
    channel_multiplier = rhs_shape[-1]

    new_rhs_shape = rhs.shape[:spatial_dims] + (1, in_channels * channel_multiplier)
    rhs_reshaped = reshape(rhs, new_rhs_shape)

    return rhs_reshaped, config_obj


def atrous_conv2d(
    value: OpValue,
    filters: OpValue,
    rate: OpValue,
    padding: OpValue,
    name: OpValue = None,
) -> OpValue:
    """Atrous convolution.

    Args:
        value (OpValue): Input `value` argument (value).
        filters (OpValue): Input `filters` argument (filters).
        rate (OpValue): Input `rate` argument (rate).
        padding (OpValue): Input `padding` argument (padding).
        name (OpValue): Input `name` argument (name).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    conv2d = get_op("Conv2D")()

    return conv2d(value, filters, strides=1, padding=padding, dilation_rate=rate)


def atrous_conv2d_transpose(
    value: OpValue,
    filters: OpValue,
    output_shape: OpValue,
    config: GenericConvConfig | None = None,
    name: OpValue = None,
) -> OpValue:
    """Atrous convolution transpose.

    Args:
        value (OpValue): Input `value` argument (value).
        filters (OpValue): Input `filters` argument (filters).
        output_shape (OpValue): Input `output_shape` argument (output shape).
        config (GenericConvConfig | None): Input `config` argument (config).
        name (OpValue): Input `name` argument (name).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    conf = config if config is not None else GenericConvConfig()

    conv2d_transpose = get_op("Conv2dTranspose")()

    return conv2d_transpose(value, filters, strides=1, padding=conf.padding, dilation_rate=conf.dilation_rate)


def bias_add(value: OpValue, bias: OpValue, data_format: OpValue = None, name: OpValue = None) -> OpValue:
    """Add `bias` to `value`.

    Args:
        value (OpValue): Input `value` argument (value).
        bias (OpValue): Input `bias` argument (bias).
        data_format (OpValue): Input `data_format` argument (data format).
        name (OpValue): Input `name` argument (name).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return add(value, bias)


def collapse_repeated(labels: OpValue, seq_length: OpValue, name: OpValue = None) -> OpValue:
    """Merge repeated labels into single labels.

    Args:
        labels (OpValue): Input `labels` argument (labels).
        seq_length (OpValue): Input `seq_length` argument (seq length).
        name (OpValue): Input `name` argument (name).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("CollapseRepeated", labels, seq_length=seq_length, name=name)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    attrs = {"seq_length": seq_length, "name": name}
    lbl_shape = extract_shape(labels)
    out = _emit_shape_node("CollapseRepeated", [labels], attrs, lbl_shape, extract_dtype(labels, "int32"))
    new_seq_len = _emit_shape_node("CollapseRepeatedSeqLength", [labels], attrs, lbl_shape, "int32")
    return out, new_seq_len


def compute_average_loss(
    per_example_loss: OpValue, sample_weight: OpValue = None, global_batch_size: OpValue = None
) -> OpValue:
    """Compute the average loss.

    Args:
        per_example_loss (OpValue): Input `per_example_loss` argument (per example loss).
        sample_weight (OpValue): Input `sample_weight` argument (sample weight).
        global_batch_size (OpValue): Input `global_batch_size` argument (global batch size).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    if sample_weight is not None:
        per_example_loss = multiply(per_example_loss, sample_weight)
    return mean(per_example_loss)


def depthwise_conv2d(
    input: Tensor,
    filter: Tensor,
    config: GenericConvConfig | Sequence[int] | None = None,
    name: str | None = None,
) -> OpValue:
    """Depthwise 2-D convolution.

    Args:
        input (Tensor): Input tensor.
        filter (Tensor): Depthwise convolution filter tensor.
        config (GenericConvConfig | Sequence[int] | None): Convolution configuration or strides.
        name (str | None): Optional op name.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    conf = (
        config
        if isinstance(config, GenericConvConfig)
        else (GenericConvConfig(strides=config) if isinstance(config, (tuple, list)) else GenericConvConfig())
    )

    conv2d = get_op("Conv2d")()

    return conv2d(
        input,
        filter,
        strides=conf.strides,
        padding=conf.padding,
        dilation_rate=conf.dilation_rate,
        groups=filter.shape[2],
    )


def _resolve_backprop_target_shape(sizes: OpValue) -> tuple[int, ...]:
    """Resolve backprop target shape from an integer, sequence, or shaped tensor.

    Args:
        sizes (OpValue): Shape descriptor.

    Returns:
        tuple[int, ...]: Resolved integer shape tuple.
    """
    if isinstance(sizes, int):
        return (int(sizes),)
    if isinstance(sizes, (list, tuple)):
        return tuple(coerce_int(d) for d in sizes)
    return extract_shape(sizes) if isinstance(sizes, ShapeProtocol) else ()


def depthwise_conv2d_backprop_filter(
    input: OpValue,
    filter_sizes: OpValue,
    out_backprop: OpValue,
    config: GenericConvConfig | Sequence[int] | None = None,
    name: OpValue = None,
) -> OpValue:
    """Compute the gradients of depthwise convolution with respect to the filter.

    Args:
        input (OpValue): Input activation tensor.
        filter_sizes (OpValue): Shape tuple of the filter tensor.
        out_backprop (OpValue): Upstream gradient tensor with respect to convolution output.
        config (GenericConvConfig | Sequence[int] | None): Optional convolution configuration or strides.
        name (OpValue): Optional operation name.

    Returns:
        OpValue: Filter gradient tensor of shape `filter_sizes`.
    """
    from ml_switcheroo_compiler.core.config import config as core_config

    target_shape = _resolve_backprop_target_shape(filter_sizes)
    dtype = extract_dtype(input, "float32")
    if core_config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op(
            "DepthwiseConv2dNativeBackpropFilter",
            extract_data(input),
            filter_sizes,
            extract_data(out_backprop),
            config=config,
            name=name,
        )
        return Tensor(data, TensorConfig(target_shape, dtype, extract_device(input, "cpu")))
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node(
        "DepthwiseConv2dNativeBackpropFilter",
        [input, filter_sizes, out_backprop],
        {"config": config, "name": name},
        target_shape,
        dtype,
    )


def depthwise_conv2d_backprop_input(
    input_sizes: OpValue,
    filter: OpValue,
    out_backprop: OpValue,
    config: GenericConvConfig | Sequence[int] | None = None,
    name: OpValue = None,
) -> OpValue:
    """Compute the gradients of depthwise convolution with respect to the input.

    Args:
        input_sizes (OpValue): Shape tuple of the input tensor.
        filter (OpValue): Depthwise filter tensor.
        out_backprop (OpValue): Upstream gradient tensor with respect to convolution output.
        config (GenericConvConfig | Sequence[int] | None): Optional convolution configuration or strides.
        name (OpValue): Optional operation name.

    Returns:
        OpValue: Input gradient tensor of shape `input_sizes`.
    """
    from ml_switcheroo_compiler.core.config import config as core_config

    target_shape = _resolve_backprop_target_shape(input_sizes)
    dtype = extract_dtype(filter, "float32")
    if core_config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op(
            "DepthwiseConv2dNativeBackpropInput",
            input_sizes,
            extract_data(filter),
            extract_data(out_backprop),
            config=config,
            name=name,
        )
        return Tensor(data, TensorConfig(target_shape, dtype, extract_device(filter, "cpu")))
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node(
        "DepthwiseConv2dNativeBackpropInput",
        [input_sizes, filter, out_backprop],
        {"config": config, "name": name},
        target_shape,
        dtype,
    )


class ConvSpatialArgs(pydantic.BaseModel):
    """Arguments for spatial convolutions and morphological ops."""

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    strides: OpValue = None
    padding: str = "VALID"
    data_format: str | None = None
    dilations: OpValue = None
    rates: OpValue = None
    name: str | None = None


def dilation2d(
    input: OpValue,
    filter: OpValue,
    args: ConvSpatialArgs | None = None,
) -> OpValue:
    """Compute the grayscale dilation of 4-D `input` and 3-D `filter` tensors.

    Args:
        input (OpValue): Input `input` argument (input).
        filter (OpValue): Input `filter` argument (filter).
        args (ConvSpatialArgs | None): Input `args` argument (args).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Dilation2d", input, filter, args=args)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node(
        "Dilation2d", [input, filter], {"args": args}, extract_shape(input), extract_dtype(input, "float32")
    )


def erosion2d(
    value: OpValue,
    kernel: OpValue,
    args: ConvSpatialArgs | None = None,
) -> OpValue:
    """Compute the grayscale erosion of 4-D `value` and 3-D `kernel` tensors.

    Args:
        value (OpValue): Input `value` argument (value).
        kernel (OpValue): Input `kernel` argument (kernel).
        args (ConvSpatialArgs | None): Input `args` argument (args).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Erosion2d", value, kernel, args=args)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node(
        "Erosion2d", [value, kernel], {"args": args}, extract_shape(value), extract_dtype(value, "float32")
    )


def convolution(
    input: OpValue,
    filters: OpValue,
    args: ConvSpatialArgs | None = None,
) -> OpValue:
    """Compute sums of N-D convolutions (actually cross-correlation).

    Args:
        input (OpValue): Input `input` argument (input).
        filters (OpValue): Input `filters` argument (filters).
        args (ConvSpatialArgs | None): Input `args` argument (args).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    args = args or ConvSpatialArgs()
    _conv_nd = get_op("_ConvNd")()

    num_spatial_dims = len(extract_shape(input)) - 2
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
    input: OpValue,
    filters: OpValue,
    output_shape: OpValue,
    args: ConvSpatialArgs | None = None,
) -> OpValue:
    """Return the transpose of `convolution`.

    Args:
        input (OpValue): Input `input` argument (input).
        filters (OpValue): Input `filters` argument (filters).
        output_shape (OpValue): Input `output_shape` argument (output shape).
        args (ConvSpatialArgs | None): Input `args` argument (args).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("ConvTranspose", input, filters, output_shape=output_shape, args=args)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node(
        "ConvTranspose",
        [input, filters],
        {"output_shape": output_shape, "args": args},
        output_shape,
        extract_dtype(input, "float32"),
    )
