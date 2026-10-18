"""Core abstractions and logic definitions for conv.py."""

from __future__ import annotations

import typing  # noqa: F401
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, cast

from ml_switcheroo_compiler.backends.registry import EagerTensorProtocol
from ml_switcheroo_compiler.core.shape import TensorOrScalar
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

    window_strides: Sequence[int] | None = None
    padding: str | Sequence[tuple[int, int]] | None = None
    filter_shape: Sequence[int] | None = None


@dispatch_eager("ConvGeneralDilated")
def conv_general_dilated(
    lhs: Tensor,
    rhs: Tensor,
    config: ConvConfig,
) -> TensorOrScalar:
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

    return cast(TensorOrScalar, cast("Callable[..., EagerTensorProtocol]", _emit_linalg_node)("ConvGeneralDilated", inputs, attributes, [out_shape], [lhs.dtype]))


@dispatch_eager("ConvGeneralDilatedLocal")
def conv_general_dilated_local(
    lhs: Tensor,
    rhs: Tensor,
    config: ConvLocalHyperparams,
    **kwargs: TensorOrScalar,
) -> TensorOrScalar:
    """ConvGeneralDilatedLocal.

    Args:
        lhs (Tensor): The lhs parameter.
        rhs (Tensor): The rhs parameter.
        config (ConvLocalHyperparams): The config parameter.
        **kwargs (TensorOrScalar): Keyword args.

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

    out_shape = getattr(ConvGeneralDilatedLocal(), "infer_shape")(lhs, rhs, **attributes)
    return _emit_linalg_node("ConvGeneralDilatedLocal", inputs, attributes, [out_shape], [lhs.dtype])


@dispatch_eager("ConvGeneralDilatedPatches")
def conv_general_dilated_patches(lhs: Tensor, filter_shape: TensorOrScalar, window_strides: TensorOrScalar, padding: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """ConvGeneralDilatedPatches.

    Args:
        lhs (Tensor): The lhs parameter.
        filter_shape (TensorOrScalar): The filter_shape parameter.
        window_strides (TensorOrScalar): The window_strides parameter.
        padding (TensorOrScalar): The padding parameter.
        **kwargs (TensorOrScalar): Keyword args.

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

    out_shape = getattr(ConvGeneralDilatedPatches(), "infer_shape")(lhs, **attributes)
    return _emit_linalg_node("ConvGeneralDilatedPatches", inputs, attributes, [out_shape], [lhs.dtype])


@dispatch_eager("ConvWithGeneralPadding")
def conv_with_general_padding(lhs: Tensor, rhs: Tensor, window_strides: TensorOrScalar, padding: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """ConvWithGeneralPadding.

    Args:
        lhs (Tensor): The lhs parameter.
        rhs (Tensor): The rhs parameter.
        window_strides (TensorOrScalar): The window_strides parameter.
        padding (TensorOrScalar): The padding parameter.
        **kwargs (TensorOrScalar): Keyword args.

    Returns:
        Tensor: Result.
    """
    inputs = [lhs, rhs]
    attributes = {"window_strides": window_strides, "padding": padding, **kwargs}

    out_shape = getattr(ConvWithGeneralPadding(), "infer_shape")(lhs, rhs, **attributes)
    return _emit_linalg_node("ConvWithGeneralPadding", inputs, attributes, [out_shape], [lhs.dtype])
