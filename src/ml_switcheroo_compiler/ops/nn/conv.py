"""Canonical 1D, 2D, 3D, depthwise, separable, transposed, and N-D neural convolution operations.

(`ml_switcheroo_compiler.ops.nn.conv`).
"""

from __future__ import annotations

import typing
from collections.abc import Sequence

import pydantic

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.constants import MATRIX_RANK, MIN_CONV_RANK
from ml_switcheroo_compiler.core.shape import OpValue, extract_data, extract_shape
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig, ensure_tensor
from ml_switcheroo_compiler.ops.configs import ConvConfig
from ml_switcheroo_compiler.ops.linalg.conv import conv_general_dilated
from ml_switcheroo_compiler.ops.linalg.conv import conv_general_dilated as conv_general
from ml_switcheroo_compiler.ops.linalg.utils import _emit_linalg_node
from ml_switcheroo_compiler.ops.registry import get_op

from .conv_utils import GenericConvConfig, _build_conv_config, _prepare_depthwise_conv


@typing.runtime_checkable
class _ConvHyperparamsProtocol(typing.Protocol):
    """Structural protocol for convolution hyperparameter objects exposing `strides` and `padding`."""

    strides: Sequence[int] | int
    padding: str | Sequence[tuple[int, int]]


class ConvHyperparams(pydantic.BaseModel):
    """Hyperparameters for separable 1D and 2D convolution operations."""

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    strides: Sequence[int] | int = 1
    padding: str | Sequence[tuple[int, int]] = "VALID"


def _conv_fixed_rank(
    lhs: Tensor,
    rhs: Tensor,
    dimension_numbers: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    config_obj: ConvConfig | OpValue = None,
    **kwargs: OpValue,
) -> OpValue:
    """Dispatch a fixed-rank channels-last convolution through `conv_general_dilated`.

    Args:
        lhs (Tensor): Channels-last input tensor.
        rhs (Tensor): Spatial filter tensor.
        dimension_numbers (tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]): Layout permutation spec.
        config_obj (ConvConfig | OpValue): Optional prebuilt `ConvConfig`.
        **kwargs (OpValue): Optional stride, padding, and dilation keyword arguments.

    Returns:
        OpValue: Convolved output tensor.
    """
    if config_obj is None:
        config_obj = _build_conv_config(kwargs, dimension_numbers)
    assert isinstance(config_obj, ConvConfig)
    return conv_general_dilated(lhs, rhs, config_obj)


def _conv_transpose_op(
    lhs: Tensor,
    rhs: Tensor,
    strides: Sequence[int] | int = 1,
    padding: str | Sequence[tuple[int, int]] = "VALID",
) -> OpValue:
    """Invoke the registered `ConvTranspose` operator on `(lhs, rhs)`.

    Args:
        lhs (Tensor): Input activation tensor.
        rhs (Tensor): Filter tensor.
        strides (Sequence[int] | int): Transposed convolution strides.
        padding (str | Sequence[tuple[int, int]]): Padding mode or explicit pairs.

    Returns:
        OpValue: Transposed convolution result.
    """
    conv_transpose_def = get_op("ConvTranspose")()
    return conv_transpose_def(lhs, rhs, strides, padding)


def conv1d(
    lhs: Tensor,
    rhs: Tensor,
    config_obj: ConvConfig | OpValue = None,
    **kwargs: OpValue,
) -> OpValue:
    """Compute 1D convolution over `(batch, length, in_channels)` inputs.

    Args:
        lhs (Tensor): Input tensor of shape `(batch, length, in_channels)`.
        rhs (Tensor): Filter tensor of shape `(kernel_length, in_channels, out_channels)`.
        config_obj (ConvConfig | OpValue): Optional explicit convolution configuration.
        **kwargs (OpValue): Additional stride, padding, or dilation keyword arguments.

    Returns:
        OpValue: 1D convolution output tensor.
    """
    return _conv_fixed_rank(lhs, rhs, ((0, 2, 1), (2, 1, 0), (0, 2, 1)), config_obj, **kwargs)


def conv1d_transpose(
    lhs: Tensor,
    rhs: Tensor,
    strides: Sequence[int] | int = 1,
    padding: str | Sequence[tuple[int, int]] = "VALID",
) -> OpValue:
    """Compute 1D transposed convolution.

    Args:
        lhs (Tensor): Input tensor.
        rhs (Tensor): Kernel tensor.
        strides (Sequence[int] | int): Spatial stride(s).
        padding (str | Sequence[tuple[int, int]]): Padding specification.

    Returns:
        OpValue: 1D transposed convolution output tensor.
    """
    return _conv_transpose_op(lhs, rhs, strides, padding)


def depthwise_conv1d(
    lhs: Tensor,
    rhs: Tensor,
    config_obj: ConvConfig | OpValue = None,
    **kwargs: OpValue,
) -> OpValue:
    """Compute 1D depthwise convolution.

    Args:
        lhs (Tensor): Input tensor.
        rhs (Tensor): Depthwise kernel tensor.
        config_obj (ConvConfig | OpValue): Optional explicit convolution configuration.
        **kwargs (OpValue): Additional stride, padding, or dilation keyword arguments.

    Returns:
        OpValue: 1D depthwise convolution output tensor.
    """
    dimension_numbers = ((0, 2, 1), (2, 1, 0), (0, 2, 1))
    rhs_reshaped, resolved_cfg = _prepare_depthwise_conv(lhs, rhs, 1, dimension_numbers, config_obj, **kwargs)
    assert isinstance(resolved_cfg, ConvConfig)
    return conv_general_dilated(lhs, ensure_tensor(rhs_reshaped), resolved_cfg)


def separable_conv1d(
    lhs: Tensor,
    depthwise_filter: Tensor,
    pointwise_filter: Tensor,
    config: ConvHyperparams | OpValue = None,
    **kwargs: OpValue,
) -> OpValue:
    """Compute 1D depthwise-separable convolution (`depthwise_conv1d` followed by `1x1` `conv1d`).

    Args:
        lhs (Tensor): Input tensor.
        depthwise_filter (Tensor): Depthwise spatial filter tensor.
        pointwise_filter (Tensor): Pointwise channel-mixing filter tensor.
        config (ConvHyperparams | OpValue): Optional separable convolution hyperparameters.
        **kwargs (OpValue): Additional convolution keyword arguments.

    Returns:
        OpValue: 1D separable convolution output tensor.
    """
    cfg = config if isinstance(config, (ConvHyperparams, _ConvHyperparamsProtocol)) else ConvHyperparams()
    kwargs["strides"] = cfg.strides
    kwargs["padding"] = cfg.padding
    depthwise_out = depthwise_conv1d(lhs, depthwise_filter, None, **kwargs)
    return conv1d(ensure_tensor(depthwise_out), pointwise_filter, None, strides=1, padding="VALID")


def conv2d(
    lhs: Tensor,
    rhs: Tensor,
    config_obj: ConvConfig | OpValue = None,
    **kwargs: OpValue,
) -> OpValue:
    """Compute 2D convolution over `(batch, height, width, in_channels)` inputs.

    Args:
        lhs (Tensor): Input tensor of shape `(batch, height, width, in_channels)`.
        rhs (Tensor): Filter tensor of shape `(k_h, k_w, in_channels, out_channels)`.
        config_obj (ConvConfig | OpValue): Optional explicit convolution configuration.
        **kwargs (OpValue): Additional stride, padding, or dilation keyword arguments.

    Returns:
        OpValue: 2D convolution output tensor.
    """
    return _conv_fixed_rank(lhs, rhs, ((0, 3, 1, 2), (3, 2, 0, 1), (0, 3, 1, 2)), config_obj, **kwargs)


def conv2d_transpose(
    lhs: Tensor,
    rhs: Tensor,
    strides: Sequence[int] | int = 1,
    padding: str | Sequence[tuple[int, int]] = "VALID",
) -> OpValue:
    """Compute 2D transposed convolution.

    Args:
        lhs (Tensor): Input tensor.
        rhs (Tensor): Kernel tensor.
        strides (Sequence[int] | int): Spatial stride(s).
        padding (str | Sequence[tuple[int, int]]): Padding specification.

    Returns:
        OpValue: 2D transposed convolution output tensor.
    """
    return _conv_transpose_op(lhs, rhs, strides, padding)


def depthwise_conv2d(
    lhs: Tensor,
    rhs: Tensor,
    config_obj: ConvConfig | OpValue = None,
    **kwargs: OpValue,
) -> OpValue:
    """Compute 2D depthwise convolution.

    Args:
        lhs (Tensor): Input tensor.
        rhs (Tensor): Depthwise kernel tensor.
        config_obj (ConvConfig | OpValue): Optional explicit convolution configuration.
        **kwargs (OpValue): Additional stride, padding, or dilation keyword arguments.

    Returns:
        OpValue: 2D depthwise convolution output tensor.
    """
    dimension_numbers = ((0, 3, 1, 2), (3, 2, 0, 1), (0, 3, 1, 2))
    rhs_reshaped, resolved_cfg = _prepare_depthwise_conv(lhs, rhs, 2, dimension_numbers, config_obj, **kwargs)
    assert isinstance(resolved_cfg, ConvConfig)
    return conv_general_dilated(lhs, ensure_tensor(rhs_reshaped), resolved_cfg)


def separable_conv2d(
    lhs: Tensor,
    depthwise_filter: Tensor,
    pointwise_filter: Tensor,
    config: ConvHyperparams | OpValue = None,
    **kwargs: OpValue,
) -> OpValue:
    """Compute 2D depthwise-separable convolution (`depthwise_conv2d` followed by `1x1` `conv2d`).

    Args:
        lhs (Tensor): Input tensor.
        depthwise_filter (Tensor): Depthwise spatial filter tensor.
        pointwise_filter (Tensor): Pointwise channel-mixing filter tensor.
        config (ConvHyperparams | OpValue): Optional separable convolution hyperparameters.
        **kwargs (OpValue): Additional convolution keyword arguments.

    Returns:
        OpValue: 2D separable convolution output tensor.
    """
    cfg = config if isinstance(config, (ConvHyperparams, _ConvHyperparamsProtocol)) else ConvHyperparams()
    kwargs["strides"] = cfg.strides
    kwargs["padding"] = cfg.padding
    depthwise_out = depthwise_conv2d(lhs, depthwise_filter, None, **kwargs)
    return conv2d(ensure_tensor(depthwise_out), pointwise_filter, None, strides=1, padding="VALID")


def conv3d(
    lhs: Tensor,
    rhs: Tensor,
    config_obj: ConvConfig | OpValue = None,
    **kwargs: OpValue,
) -> OpValue:
    """Compute 3D convolution over `(batch, depth, height, width, in_channels)` inputs.

    Args:
        lhs (Tensor): Input tensor of shape `(batch, depth, height, width, in_channels)`.
        rhs (Tensor): Filter tensor of shape `(k_d, k_h, k_w, in_channels, out_channels)`.
        config_obj (ConvConfig | OpValue): Optional explicit convolution configuration.
        **kwargs (OpValue): Additional stride, padding, or dilation keyword arguments.

    Returns:
        OpValue: 3D convolution output tensor.
    """
    return _conv_fixed_rank(lhs, rhs, ((0, 4, 1, 2, 3), (4, 3, 0, 1, 2), (0, 4, 1, 2, 3)), config_obj, **kwargs)


def conv3d_transpose(
    lhs: Tensor,
    rhs: Tensor,
    strides: Sequence[int] | int = 1,
    padding: str | Sequence[tuple[int, int]] = "VALID",
) -> OpValue:
    """Compute 3D transposed convolution.

    Args:
        lhs (Tensor): Input tensor.
        rhs (Tensor): Kernel tensor.
        strides (Sequence[int] | int): Spatial stride(s).
        padding (str | Sequence[tuple[int, int]]): Padding specification.

    Returns:
        OpValue: 3D transposed convolution output tensor.
    """
    return _conv_transpose_op(lhs, rhs, strides, padding)


def conv_transpose(
    lhs: Tensor,
    rhs: Tensor,
    strides: Sequence[int] | int = 1,
    padding: str | Sequence[tuple[int, int]] = "VALID",
) -> OpValue:
    """Compute generic N-D transposed convolution in eager or symbolic tracing mode.

    Args:
        lhs (Tensor): Left-hand input tensor.
        rhs (Tensor): Right-hand kernel tensor.
        strides (Sequence[int] | int): Spatial stride(s).
        padding (str | Sequence[tuple[int, int]]): Padding specification.

    Returns:
        OpValue: Transposed convolution output tensor.
    """
    op_name = "ConvTranspose"

    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op(
            op_name,
            extract_data(lhs),
            extract_data(rhs),
            strides=strides,
            padding=padding,
        )
        return Tensor(data, TensorConfig(extract_shape(data), lhs.dtype, lhs.device))

    return _emit_linalg_node(op_name, [lhs, rhs], {"strides": strides, "padding": padding}, [()], [lhs.dtype])


def conv(
    inputs: Tensor,
    kernel: Tensor,
    config: GenericConvConfig | None = None,
) -> Tensor:
    """Dispatch rank-polymorphic 1D, 2D, or 3D convolution based on `inputs.shape`.

    Args:
        inputs (Tensor): Input activation tensor of rank 3, 4, or 5.
        kernel (Tensor): Convolution kernel tensor.
        config (GenericConvConfig | None): Optional generic convolution configuration.

    Returns:
        Tensor: Convolved output tensor.

    Raises:
        ValueError: If `len(inputs.shape) - 2` is not 1, 2, or 3.
    """
    conf = config if config is not None else GenericConvConfig()
    spatial_rank = len(inputs.shape) - 2
    if spatial_rank == 1:
        return ensure_tensor(
            conv1d(
                inputs,
                kernel,
                strides=conf.strides,
                padding=conf.padding,
                lhs_dilation=1,
                rhs_dilation=conf.dilation_rate,
            )
        )
    if spatial_rank == MATRIX_RANK:
        return ensure_tensor(
            conv2d(
                inputs,
                kernel,
                strides=conf.strides,
                padding=conf.padding,
                lhs_dilation=1,
                rhs_dilation=conf.dilation_rate,
            )
        )
    if spatial_rank == MIN_CONV_RANK:
        return ensure_tensor(
            conv3d(
                inputs,
                kernel,
                strides=conf.strides,
                padding=conf.padding,
                lhs_dilation=1,
                rhs_dilation=conf.dilation_rate,
            )
        )
    raise ValueError(f"Unsupported spatial rank: {spatial_rank}")


def depthwise_conv(
    inputs: Tensor,
    kernel: Tensor,
    config: GenericConvConfig | None = None,
) -> Tensor:
    """Dispatch rank-polymorphic 1D or 2D depthwise convolution based on `inputs.shape`.

    Args:
        inputs (Tensor): Input activation tensor of rank 3 or 4.
        kernel (Tensor): Depthwise kernel tensor.
        config (GenericConvConfig | None): Optional generic convolution configuration.

    Returns:
        Tensor: Depthwise convolved output tensor.

    Raises:
        ValueError: If `len(inputs.shape) - 2` is not 1 or 2.
    """
    conf = config if config is not None else GenericConvConfig()
    spatial_rank = len(inputs.shape) - 2
    if spatial_rank == 1:
        return ensure_tensor(
            depthwise_conv1d(
                inputs,
                kernel,
                strides=conf.strides,
                padding=conf.padding,
                lhs_dilation=1,
                rhs_dilation=conf.dilation_rate,
            )
        )
    if spatial_rank == MATRIX_RANK:
        return ensure_tensor(
            depthwise_conv2d(
                inputs,
                kernel,
                config=GenericConvConfig(
                    strides=conf.strides,
                    padding=conf.padding,
                    dilation_rate=conf.dilation_rate,
                ),
            )
        )
    raise ValueError(f"Unsupported spatial rank for depthwise_conv: {spatial_rank}")


def separable_conv(
    inputs: Tensor,
    depthwise_kernel: Tensor,
    pointwise_kernel: Tensor,
    config: GenericConvConfig | None = None,
) -> Tensor:
    """Dispatch rank-polymorphic 1D or 2D separable convolution based on `inputs.shape`.

    Args:
        inputs (Tensor): Input activation tensor of rank 3 or 4.
        depthwise_kernel (Tensor): Depthwise spatial kernel tensor.
        pointwise_kernel (Tensor): Pointwise channel-mixing kernel tensor.
        config (GenericConvConfig | None): Optional generic convolution configuration.

    Returns:
        Tensor: Separable convolution output tensor.

    Raises:
        ValueError: If `len(inputs.shape) - 2` is not 1 or 2.
    """
    conf = config if config is not None else GenericConvConfig()
    spatial_rank = len(inputs.shape) - 2
    if spatial_rank == 1:
        return ensure_tensor(
            separable_conv1d(
                inputs,
                depthwise_kernel,
                pointwise_kernel,
                strides=conf.strides,
                padding=conf.padding,
                lhs_dilation=1,
                rhs_dilation=conf.dilation_rate,
            )
        )
    if spatial_rank == MATRIX_RANK:
        return ensure_tensor(
            separable_conv2d(
                inputs,
                depthwise_kernel,
                pointwise_kernel,
                strides=conf.strides,
                padding=conf.padding,
                lhs_dilation=1,
                rhs_dilation=conf.dilation_rate,
            )
        )
    raise ValueError(f"Unsupported spatial rank for separable_conv: {spatial_rank}")


conv_transpose1d = conv1d_transpose
conv_transpose2d = conv2d_transpose
conv_transpose3d = conv3d_transpose

__all__ = [
    "ConvHyperparams",
    "GenericConvConfig",
    "conv",
    "conv1d",
    "conv1d_transpose",
    "conv2d",
    "conv2d_transpose",
    "conv3d",
    "conv3d_transpose",
    "conv_general",
    "conv_transpose",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "depthwise_conv",
    "depthwise_conv1d",
    "depthwise_conv2d",
    "separable_conv",
    "separable_conv1d",
    "separable_conv2d",
]
