# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Convolution operations."""

import math
import typing
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Union

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
) -> Sequence[tuple[int, int]]:
    """Calculate padding for transposed convolution.

    Args:
        padding (Union[str, Sequence[tuple[int, int]]]): Padding mode or sequence.
        k_sdims (tuple[int, ...]): Kernel spatial dimensions.
        strides_tuple (tuple[int, ...]): Strides tuple.

    Returns:
        Sequence[tuple[int, int]]: The calculated padding sequence.
    """
    if not (isinstance(padding, str) and padding in {"SAME", "VALID"}):
        return padding  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    pads = []
    for k, s in zip(k_sdims, strides_tuple):
        if padding == "SAME":
            pads.append(_calc_same_pad(k, s))
        else:  # VALID
            pads.append(_calc_valid_pad(k, s))
    return pads


def _build_conv_config(kwargs: dict[str, Any], dimension_numbers: tuple[Any, ...]) -> Any:
    """Evaluate _build_conv_config operation.

    Args:
        kwargs (dict): The kwargs parameter.
        dimension_numbers (tuple): The dimension_numbers parameter.

    Returns: Any: Result.
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
        window_strides=strides,
        padding=kwargs.get("padding", "VALID"),
        lhs_dilation=lhs_dilation,
        rhs_dilation=rhs_dilation,
        dimension_numbers=dimension_numbers,
    )


def _prepare_depthwise_conv(
    lhs: Tensor,  # type: ignore
    rhs: Tensor,  # type: ignore
    spatial_dims: int,
    dimension_numbers: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    config_obj: typing.Optional[Any] = None,
    **kwargs: Any,
) -> tuple[Tensor, Any]:  # type: ignore
    """Prepare configuration and reshape weights for depthwise convolution.

    Args:
        lhs (Tensor): The lhs parameter.
        rhs (Tensor): The rhs parameter.
        spatial_dims (int): The spatial_dims parameter.
        dimension_numbers (tuple): The dimension_numbers parameter.
        config_obj (object): The config_obj parameter.
        **kwargs (object): Keyword args.

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

    new_rhs_shape = rhs.shape[:spatial_dims] + (1, in_channels * channel_multiplier)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
    rhs_reshaped = reshape(rhs, new_rhs_shape)

    return rhs_reshaped, config_obj


def atrous_conv2d(value: Any, filters: Any, rate: Any, padding: Any, name: Any = None) -> Any:
    """Atrous convolution.

    Args:
        value (object): The value parameter.
        filters (object): The filters parameter.
        rate (object): The rate parameter.
        padding (object): The padding parameter.
        name (object): The name parameter.

    Returns: Any: Result.
    """
    conv2d = get_op("Conv2d")()

    return conv2d(value, filters, strides=1, padding=padding, dilation_rate=rate)


def atrous_conv2d_transpose(
    value: Any,
    filters: Any,
    output_shape: Any,
    config: typing.Optional[GenericConvConfig] = None,
    name: Any = None,
) -> Any:
    """Atrous convolution transpose.

    Args:
        value (object): The value parameter.
        filters (object): The filters parameter.
        output_shape (object): The output_shape parameter.
        config (object): The config parameter.
        name (object): The name parameter.

    Returns: Any: Result.
    """
    conf = config if config is not None else GenericConvConfig()

    conv2d_transpose = get_op("Conv2dTranspose")()

    return conv2d_transpose(value, filters, strides=1, padding=conf.padding, dilation_rate=conf.dilation_rate)


def bias_add(value: Any, bias: Any, data_format: Any = None, name: Any = None) -> Any:
    """Add `bias` to `value`.

    Args:
        value (object): The value parameter.
        bias (object): The bias parameter.
        data_format (object): The data_format parameter.
        name (object): The name parameter.

    Returns: Any: Result.
    """
    return add(value, bias)


def collapse_repeated(labels: Any, seq_length: Any, name: Any = None) -> Any:
    """Merge repeated labels into single labels.

    Args:
        labels (object): The labels parameter.
        seq_length (object): The seq_length parameter.
        name (object): The name parameter.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("CollapseRepeated", labels, seq_length=seq_length, name=name)
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    out = _emit_shape_node("CollapseRepeated", [labels], {"seq_length": seq_length, "name": name}, getattr(labels, "shape", ()), getattr(labels, "dtype", "int32"))
    return out, Tensor(None, TensorConfig(getattr(labels, "shape", ()), "int32", "cpu"))


def compute_average_loss(per_example_loss: Any, sample_weight: Any = None, global_batch_size: Any = None) -> Any:
    """Compute the average loss.

    Args:
        per_example_loss (object): The per_example_loss parameter.
        sample_weight (object): The sample_weight parameter.
        global_batch_size (object): The global_batch_size parameter.

    Returns: Any: Result.
    """
    if sample_weight is not None:
        per_example_loss = multiply(per_example_loss, sample_weight)
    return mean(per_example_loss)


def depthwise_conv2d(
    input: Tensor,  # type: ignore
    filter: Tensor,  # type: ignore
    config: typing.Optional[GenericConvConfig] = None,
    name: typing.Optional[str] = None,
) -> Any:
    """Depthwise 2-D convolution.

    Args:
        input (Tensor): The input parameter.
        filter (Tensor): The filter parameter.
        config (object): The config parameter.
        name (object): The name parameter.

    Returns: Any: Result.
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
    input: Any,
    filter_sizes: Any,
    out_backprop: Any,
    config: typing.Optional[GenericConvConfig] = None,
    name: Any = None,
) -> Any:
    """Compute the gradients of depthwise convolution with respect to the filter.

    Args:
        input (object): The input parameter.
        filter_sizes (object): The filter_sizes parameter.
        out_backprop (object): The out_backprop parameter.
        config (object): The config parameter.
        name (object): The name parameter.

    Returns: Any: Result.
    """
    return Tensor(None, TensorConfig(filter_sizes, "float32", "cpu"))


def depthwise_conv2d_backprop_input(
    input_sizes: Any,
    filter: Any,
    out_backprop: Any,
    config: typing.Optional[GenericConvConfig] = None,
    name: Any = None,
) -> Any:
    """Compute the gradients of depthwise convolution with respect to the input.

    Args:
        input_sizes (object): The input_sizes parameter.
        filter (object): The filter parameter.
        out_backprop (object): The out_backprop parameter.
        config (object): The config parameter.
        name (object): The name parameter.

    Returns: Any: Result.
    """
    return Tensor(None, TensorConfig(input_sizes, "float32", "cpu"))


@dataclass
class ConvSpatialArgs:
    """Arguments for spatial convolutions and morphological ops."""

    strides: typing.Optional[Any] = None
    padding: Any = "VALID"
    data_format: typing.Optional[Any] = None
    dilations: typing.Optional[Any] = None
    rates: typing.Optional[Any] = None
    name: typing.Optional[Any] = None


def dilation2d(
    input: Any,
    filter: Any,
    args: typing.Optional[ConvSpatialArgs] = None,
) -> Any:
    """Compute the grayscale dilation of 4-D `input` and 3-D `filter` tensors.

    Args:
        input (object): The input parameter.
        filter (object): The filter parameter.
        args (object): The args parameter.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Dilation2d", input, filter, args=args)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Dilation2d", [input, filter], {"args": args}, getattr(input, "shape", ()), getattr(input, "dtype", "float32"))


def erosion2d(
    value: Any,
    kernel: Any,
    args: typing.Optional[ConvSpatialArgs] = None,
) -> Any:
    """Compute the grayscale erosion of 4-D `value` and 3-D `kernel` tensors.

    Args:
        value (object): The value parameter.
        kernel (object): The kernel parameter.
        args (object): The args parameter.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Erosion2d", value, kernel, args=args)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Erosion2d", [value, kernel], {"args": args}, getattr(value, "shape", ()), getattr(value, "dtype", "float32"))


def convolution(
    input: Any,
    filters: Any,
    args: typing.Optional[ConvSpatialArgs] = None,
) -> Any:
    """Compute sums of N-D convolutions (actually cross-correlation).

    Args:
        input (object): The input parameter.
        filters (object): The filters parameter.
        args (object): The args parameter.

    Returns: Any: Result.
    """
    args = args or ConvSpatialArgs()
    _conv_nd = get_op("_ConvNd")()

    num_spatial_dims = len(input.shape) - 2
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
    input: Any,
    filters: Any,
    output_shape: Any,
    args: typing.Optional[ConvSpatialArgs] = None,
) -> Any:
    """Return the transpose of `convolution`.

    Args:
        input (object): The input parameter.
        filters (object): The filters parameter.
        output_shape (object): The output_shape parameter.
        args (object): The args parameter.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("ConvTranspose", input, filters, output_shape=output_shape, args=args)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("ConvTranspose", [input, filters], {"output_shape": output_shape, "args": args}, output_shape, getattr(input, "dtype", "float32"))
