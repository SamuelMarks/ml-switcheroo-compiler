"""Core abstractions and logic definitions for conv.py."""

from __future__ import annotations

from dataclasses import dataclass

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.base import dispatch_eager
from ml_switcheroo_compiler.ops.configs import ConvConfig
from ml_switcheroo_compiler.ops.linalg.conv_ops import (
    ConvGeneralDilated,
    ConvGeneralDilatedLocal,
    ConvGeneralDilatedPatches,
    ConvWithGeneralPadding,
)
from ml_switcheroo_compiler.ops.linalg.utils import _emit_linalg_node


@dataclass
class ConvLocalHyperparams:
    """ConvLocalHyperparams."""

    window_strides: object
    padding: object
    filter_shape: object


@dispatch_eager("ConvGeneralDilated")
def conv_general_dilated(
    lhs: Tensor,
    rhs: Tensor,
    config: ConvConfig,
) -> Tensor:
    """General N-dimensional convolution with support for strides, padding, and dilations.

    Args:
        lhs (Tensor): The lhs parameter.
        rhs (Tensor): The rhs parameter.
        config (ConvConfig): The config parameter.

    Returns:
        Tensor: Result.
    """
    inputs = [lhs, rhs]
    attributes = {
        "config": config,
    }

    op = ConvGeneralDilated()

    cfg = ConvConfig(
        config.window_strides,
        config.padding,
        config.lhs_dilation,
        config.rhs_dilation,
        config.dimension_numbers,
        config.feature_group_count,
    )
    out_shape = op.infer_shape(lhs, rhs, cfg)

    return _emit_linalg_node("ConvGeneralDilated", inputs, attributes, [out_shape], [lhs.dtype])


@dispatch_eager("ConvGeneralDilatedLocal")
def conv_general_dilated_local(
    lhs: Tensor,
    rhs: Tensor,
    config: ConvLocalHyperparams,
    **kwargs: object,
) -> Tensor:
    """ConvGeneralDilatedLocal.

    Args:
        lhs (Tensor): The lhs parameter.
        rhs (Tensor): The rhs parameter.
        config (ConvLocalHyperparams): The config parameter.
        **kwargs (object): Keyword args.

    Returns:
        Tensor: Result.
    """
    window_strides, padding, filter_shape = config.window_strides, config.padding, config.filter_shape
    inputs = [lhs, rhs]
    attributes = {
        "window_strides": window_strides,
        "padding": padding,
        "filter_shape": filter_shape,
        **kwargs,
    }

    out_shape = ConvGeneralDilatedLocal().infer_shape(lhs, rhs, **attributes)
    return _emit_linalg_node("ConvGeneralDilatedLocal", inputs, attributes, [out_shape], [lhs.dtype])


@dispatch_eager("ConvGeneralDilatedPatches")
def conv_general_dilated_patches(lhs: Tensor, filter_shape: object, window_strides: object, padding: object, **kwargs: object) -> Tensor:
    """ConvGeneralDilatedPatches.

    Args:
        lhs (Tensor): The lhs parameter.
        filter_shape (object): The filter_shape parameter.
        window_strides (object): The window_strides parameter.
        padding (object): The padding parameter.
        **kwargs (object): Keyword args.

    Returns:
        Tensor: Result.
    """
    inputs = [lhs]
    attributes = {
        "filter_shape": filter_shape,
        "window_strides": window_strides,
        "padding": padding,
        **kwargs,
    }

    out_shape = ConvGeneralDilatedPatches().infer_shape(lhs, **attributes)
    return _emit_linalg_node("ConvGeneralDilatedPatches", inputs, attributes, [out_shape], [lhs.dtype])


@dispatch_eager("ConvWithGeneralPadding")
def conv_with_general_padding(lhs: Tensor, rhs: Tensor, window_strides: object, padding: object, **kwargs: object) -> Tensor:
    """ConvWithGeneralPadding.

    Args:
        lhs (Tensor): The lhs parameter.
        rhs (Tensor): The rhs parameter.
        window_strides (object): The window_strides parameter.
        padding (object): The padding parameter.
        **kwargs (object): Keyword args.

    Returns:
        Tensor: Result.
    """
    inputs = [lhs, rhs]
    attributes = {"window_strides": window_strides, "padding": padding, **kwargs}

    out_shape = ConvWithGeneralPadding().infer_shape(lhs, rhs, **attributes)
    return _emit_linalg_node("ConvWithGeneralPadding", inputs, attributes, [out_shape], [lhs.dtype])
