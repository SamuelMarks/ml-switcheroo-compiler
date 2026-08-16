"""Module conv.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Core abstractions and logic definitions for conv.py."""
from dataclasses import dataclass
from typing import Any

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

    window_strides: Any
    padding: Any
    filter_shape: Any


@dispatch_eager("ConvGeneralDilated")
def conv_general_dilated(
    lhs: Tensor,  # type: ignore
    rhs: Tensor,  # type: ignore
    config: ConvConfig,
) -> Any:
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
    lhs: Tensor,  # type: ignore
    rhs: Tensor,  # type: ignore
    config: ConvLocalHyperparams,
    **kwargs: Any,
) -> Any:
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
def conv_general_dilated_patches(lhs: Tensor, filter_shape: Any, window_strides: Any, padding: Any, **kwargs: Any) -> Any:  # type: ignore
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
def conv_with_general_padding(lhs: Tensor, rhs: Tensor, window_strides: Any, padding: Any, **kwargs: Any) -> Any:  # type: ignore
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
