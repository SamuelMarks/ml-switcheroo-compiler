"""Linear algebra operations."""

from __future__ import annotations


from typing import TYPE_CHECKING

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.base import dispatch_eager
from ml_switcheroo_compiler.ops.linalg.frontend import _emit_linalg_node

if TYPE_CHECKING:
    pass


from ml_switcheroo_compiler.ops.configs import ConvConfig


@dispatch_eager("ConvGeneralDilated")
def conv_general_dilated(
    lhs: Tensor,
    rhs: Tensor,
    config: ConvConfig,
) -> Tensor:
    """General N-dimensional convolution with support for strides, padding, and dilations.

    Args:
        lhs (Tensor): Left-hand side tensor (input).
        rhs (Tensor): Right-hand side tensor (filters/weights).
        config (ConvConfig): The configuration for the convolution.

    Returns:
    Tensor: The result of the convolution.
    """
    inputs = [lhs, rhs]
    attributes = {
        "config": config,
    }

    from ml_switcheroo_compiler.ops.linalg.basic import ConvGeneralDilated

    op = ConvGeneralDilated()
    from ml_switcheroo_compiler.ops.linalg.basic import ConvConfig

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
    window_strides: object,
    padding: object,
    filter_shape: object,
    **kwargs: object,
) -> Tensor:
    """ConvGeneralDilatedLocal."""
    inputs = [lhs, rhs]
    attributes = {
        "window_strides": window_strides,
        "padding": padding,
        "filter_shape": filter_shape,
        **kwargs,
    }
    from ml_switcheroo_compiler.ops.linalg.basic import ConvGeneralDilatedLocal

    out_shape = ConvGeneralDilatedLocal().infer_shape(lhs, rhs, **attributes)
    return _emit_linalg_node(
        "ConvGeneralDilatedLocal", inputs, attributes, [out_shape], [lhs.dtype]
    )


@dispatch_eager("ConvGeneralDilatedPatches")
def conv_general_dilated_patches(
    lhs: Tensor, filter_shape: object, window_strides: object, padding: object, **kwargs: object
) -> Tensor:
    """ConvGeneralDilatedPatches."""
    inputs = [lhs]
    attributes = {
        "filter_shape": filter_shape,
        "window_strides": window_strides,
        "padding": padding,
        **kwargs,
    }
    from ml_switcheroo_compiler.ops.linalg.basic import ConvGeneralDilatedPatches

    out_shape = ConvGeneralDilatedPatches().infer_shape(lhs, **attributes)
    return _emit_linalg_node(
        "ConvGeneralDilatedPatches", inputs, attributes, [out_shape], [lhs.dtype]
    )


@dispatch_eager("ConvWithGeneralPadding")
def conv_with_general_padding(
    lhs: Tensor, rhs: Tensor, window_strides: object, padding: object, **kwargs: object
) -> Tensor:
    """ConvWithGeneralPadding."""
    inputs = [lhs, rhs]
    attributes = {"window_strides": window_strides, "padding": padding, **kwargs}
    from ml_switcheroo_compiler.ops.linalg.basic import ConvWithGeneralPadding

    out_shape = ConvWithGeneralPadding().infer_shape(lhs, rhs, **attributes)
    return _emit_linalg_node("ConvWithGeneralPadding", inputs, attributes, [out_shape], [lhs.dtype])
