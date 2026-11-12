"""Implementation of conv operations and utilities for the ops subsystem (`ml_switcheroo_compiler.ops.linalg.conv`)."""

from __future__ import annotations

import typing  # noqa: F401
from collections.abc import Sequence

import pydantic

from ml_switcheroo_compiler.backends.registry import ensure_callable, ensure_eager_tensor, ensure_shape
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


class ConvLocalHyperparams(pydantic.BaseModel):
    """ConvLocalHyperparams."""

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

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
        lhs (Tensor): Input `lhs` argument (lhs).
        rhs (Tensor): Input `rhs` argument (rhs).
        config (ConvConfig): Input `config` argument (config).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    inputs = [lhs, rhs]
    attributes = {
        "config": config,
    }

    op = ConvGeneralDilated()

    cfg = ConvConfig(
        window_strides=config.window_strides,
        padding=config.padding,
        lhs_dilation=config.lhs_dilation,
        rhs_dilation=config.rhs_dilation,
        dimension_numbers=config.dimension_numbers,
        feature_group_count=config.feature_group_count,
    )
    out_shape = op.infer_shape(lhs, rhs, cfg)

    return ensure_eager_tensor(ensure_callable(_emit_linalg_node)("ConvGeneralDilated", inputs, attributes, [out_shape], [lhs.dtype]))


@dispatch_eager("ConvGeneralDilatedLocal")
def conv_general_dilated_local(
    lhs: Tensor,
    rhs: Tensor,
    config: ConvLocalHyperparams,
    **kwargs: TensorOrScalar,
) -> TensorOrScalar:
    """ConvGeneralDilatedLocal.

    Args:
        lhs (Tensor): Input `lhs` argument (lhs).
        rhs (Tensor): Input `rhs` argument (rhs).
        config (ConvLocalHyperparams): Input `config` argument (config).
        **kwargs (TensorOrScalar): Keyword args.

    Returns:
        Tensor: Computed `Tensor` output.
    """
    window_strides, padding, filter_shape = config.window_strides, config.padding, config.filter_shape
    inputs = [lhs, rhs]
    attributes = {
        "window_strides": window_strides,
        "padding": padding,
        "filter_shape": filter_shape,
        **kwargs,
    }

    out_shape = ensure_shape(ConvGeneralDilatedLocal().infer_shape(ensure_eager_tensor(lhs), ensure_eager_tensor(rhs), **attributes))
    return _emit_linalg_node("ConvGeneralDilatedLocal", inputs, attributes, [out_shape], [lhs.dtype])


@dispatch_eager("ConvGeneralDilatedPatches")
def conv_general_dilated_patches(lhs: Tensor, filter_shape: TensorOrScalar, window_strides: TensorOrScalar, padding: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """ConvGeneralDilatedPatches.

    Args:
        lhs (Tensor): Input `lhs` argument (lhs).
        filter_shape (TensorOrScalar): Input `filter_shape` argument (filter shape).
        window_strides (TensorOrScalar): Input `window_strides` argument (window strides).
        padding (TensorOrScalar): Input `padding` argument (padding).
        **kwargs (TensorOrScalar): Keyword args.

    Returns:
        Tensor: Computed `Tensor` output.
    """
    inputs = [lhs]
    attributes = {
        "filter_shape": filter_shape,
        "window_strides": window_strides,
        "padding": padding,
        **kwargs,
    }

    out_shape = ensure_shape(ConvGeneralDilatedPatches().infer_shape(ensure_eager_tensor(lhs), **attributes))
    return _emit_linalg_node("ConvGeneralDilatedPatches", inputs, attributes, [out_shape], [lhs.dtype])


@dispatch_eager("ConvWithGeneralPadding")
def conv_with_general_padding(lhs: Tensor, rhs: Tensor, window_strides: TensorOrScalar, padding: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """ConvWithGeneralPadding.

    Args:
        lhs (Tensor): Input `lhs` argument (lhs).
        rhs (Tensor): Input `rhs` argument (rhs).
        window_strides (TensorOrScalar): Input `window_strides` argument (window strides).
        padding (TensorOrScalar): Input `padding` argument (padding).
        **kwargs (TensorOrScalar): Keyword args.

    Returns:
        Tensor: Computed `Tensor` output.
    """
    inputs = [lhs, rhs]
    attributes = {"window_strides": window_strides, "padding": padding, **kwargs}

    out_shape = ensure_shape(ConvWithGeneralPadding().infer_shape(ensure_eager_tensor(lhs), ensure_eager_tensor(rhs), **attributes))
    return _emit_linalg_node("ConvWithGeneralPadding", inputs, attributes, [out_shape], [lhs.dtype])
