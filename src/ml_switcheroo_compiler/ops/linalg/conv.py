"""General and transposed convolution operators and dispatch functions (`ml_switcheroo_compiler.ops.linalg.conv`)."""

from __future__ import annotations

from collections.abc import Sequence

import pydantic

from ml_switcheroo_compiler.backends.registry import (
    coerce_shape,
)
from ml_switcheroo_compiler.core.constants import BINARY_OP_ARITY
from ml_switcheroo_compiler.core.shape import OpValue, ShapeProtocol, extract_shape
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.base import OpArgValue, OpDef, dispatch_eager, register_op
from ml_switcheroo_compiler.ops.configs import ConvConfig
from ml_switcheroo_compiler.ops.linalg.products import _has_valid_shape
from ml_switcheroo_compiler.ops.linalg.utils import _emit_linalg_node


class ConvLocalHyperparams(pydantic.BaseModel):
    """Hyperparameters for unshared local N-dimensional dilated convolution."""

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    window_strides: Sequence[int] | None = None
    padding: str | Sequence[tuple[int, int]] | None = None
    filter_shape: Sequence[int] | None = None


@register_op("ConvGeneralDilated")
class ConvGeneralDilated(OpDef):
    """General N-dimensional dilated convolution operator."""

    op_name = "ConvGeneralDilated"

    def infer_shape(self, *args: OpArgValue, **kwargs: OpArgValue) -> tuple[int, ...]:
        """Infer the output shape for general N-dimensional dilated convolution.

        Args:
            *args (OpArgValue): Positional arguments `(lhs, rhs, config)`.
            **kwargs (OpArgValue): Optional keyword arguments (`lhs`, `rhs`, `config`).

        Returns:
            tuple[int, ...]: Inferred spatial output shape `(batch, out_channels, *spatial_out)`.
        """
        lhs = args[0] if len(args) > 0 else kwargs.get("lhs")
        rhs = args[1] if len(args) > 1 else kwargs.get("rhs")
        config: OpArgValue | ConvConfig = args[2] if len(args) > BINARY_OP_ARITY else kwargs.get("config", None)
        if not _has_valid_shape(lhs) or not _has_valid_shape(rhs):
            return ()

        lhs_shape = extract_shape(lhs)
        rhs_shape = extract_shape(rhs)
        if not lhs_shape or not rhs_shape or len(lhs_shape) < 3 or len(rhs_shape) < 3:
            return ()

        batch = lhs_shape[0]
        out_channels = rhs_shape[0]
        spatial_in = lhs_shape[2:]
        spatial_k = rhs_shape[2:]
        cfg = config if isinstance(config, ConvConfig) else ConvConfig(window_strides=[], padding=[])
        strides = cfg.window_strides or [1] * len(spatial_in)
        padding = cfg.padding or [(0, 0)] * len(spatial_in)
        lhs_dilation = cfg.lhs_dilation or [1] * len(spatial_in)
        rhs_dilation = cfg.rhs_dilation or [1] * len(spatial_in)

        out_spatial = []
        for i, (d_in, k) in enumerate(zip(spatial_in, spatial_k)):
            st = strides[i] if i < len(strides) else 1
            pad_i = padding[i] if i < len(padding) else (0, 0)
            pad_val = pad_i[0] + pad_i[1] if isinstance(pad_i, (list, tuple)) else int(pad_i) * 2
            d_dil = lhs_dilation[i] if i < len(lhs_dilation) else 1
            k_dil = rhs_dilation[i] if i < len(rhs_dilation) else 1
            effective_in = (int(d_in) - 1) * int(d_dil) + 1
            effective_k = (int(k) - 1) * int(k_dil) + 1
            dim_out = max(0, (effective_in + pad_val - effective_k) // int(st) + 1)
            out_spatial.append(dim_out)

        return (batch, out_channels, *out_spatial)


@register_op("Convolve")
class Convolve(OpDef):
    """Discrete 1D linear convolution operator."""

    op_name = "Convolve"
    np_op_name = "convolve"

    def infer_shape(
        self,
        a: OpValue,
        v: OpValue,
        mode: str = "full",
        **kwargs: OpValue,
    ) -> OpValue:
        """Infer the output shape for 1D discrete linear convolution.

        Args:
            a (OpValue): First 1D input sequence.
            v (OpValue): Second 1D filter sequence.
            mode (str): Boundary mode (`'full'`, `'same'`, or `'valid'`).
            **kwargs (OpValue): Additional keyword attributes.

        Returns:
            OpValue: Inferred 1D output shape tuple.
        """
        if not _has_valid_shape(a) or not _has_valid_shape(v):
            return ()
        a_shape = extract_shape(a)
        v_shape = extract_shape(v)
        na = a_shape[0] if a_shape else 1
        nv = v_shape[0] if v_shape else 1
        if mode == "full":
            return (int(na) + int(nv) - 1,)
        if mode == "same":
            return (max(int(na), int(nv)),)
        if mode == "valid":
            return (max(int(na), int(nv)) - min(int(na), int(nv)) + 1,)
        return (int(na) + int(nv) - 1,)


@register_op("ConvGeneralDilatedLocal")
class ConvGeneralDilatedLocal(OpDef):
    """Unshared-weights general N-dimensional dilated local convolution operator."""

    op_name = "ConvGeneralDilatedLocal"

    def infer_shape(self, *args: ShapeProtocol, **kwargs: OpValue) -> OpValue:
        """Infer output shape for local dilated convolution.

        Args:
            *args (ShapeProtocol): Positional input tensors.
            **kwargs (OpValue): Keyword attributes.

        Returns:
            OpValue: Inferred output shape tuple.
        """
        return extract_shape(args[0]) if args else ()


@register_op("ConvGeneralDilatedPatches")
class ConvGeneralDilatedPatches(OpDef):
    """Sliding-window dilated patch extraction operator."""

    op_name = "ConvGeneralDilatedPatches"

    def infer_shape(self, *args: ShapeProtocol, **kwargs: OpValue) -> OpValue:
        """Infer output shape for dilated patch extraction.

        Args:
            *args (ShapeProtocol): Positional input tensors.
            **kwargs (OpValue): Keyword attributes.

        Returns:
            OpValue: Inferred output shape tuple.
        """
        return extract_shape(args[0]) if args else ()


@register_op("ConvWithGeneralPadding")
class ConvWithGeneralPadding(OpDef):
    """General convolution with explicit asymmetric per-dimension padding."""

    op_name = "ConvWithGeneralPadding"

    def infer_shape(self, *args: ShapeProtocol, **kwargs: OpValue) -> OpValue:
        """Infer output shape for general-padded convolution.

        Args:
            *args (ShapeProtocol): Positional input tensors.
            **kwargs (OpValue): Keyword attributes.

        Returns:
            OpValue: Inferred output shape tuple.
        """
        return extract_shape(args[0]) if args else ()


@register_op("ConvTransposeShapeTuple")
class ConvTransposeShapeTuple(OpDef):
    """Transposed convolution output shape helper operator."""

    op_name = "ConvTransposeShapeTuple"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Return empty shape tuple metadata for transposed shape helper.

        Args:
            *args (OpValue): Positional input arguments.
            **kwargs (OpValue): Keyword attributes.

        Returns:
            OpValue: Empty shape tuple `()`.
        """
        return ()


@register_op("ConvTranspose")
class ConvTranspose(OpDef):
    """N-dimensional transposed convolution (fractionally strided convolution) operator."""

    op_name = "ConvTranspose"

    def infer_shape(
        self,
        lhs: OpValue,
        rhs: OpValue,
        **kwargs: OpValue,
    ) -> OpValue:
        """Infer channels-last output shape for transposed convolution.

        Args:
            lhs (OpValue): Channels-last input tensor `(batch, *spatial, c_in)`.
            rhs (OpValue): Filter tensor `(*spatial_k, c_in, c_out)`.
            **kwargs (OpValue): Keyword attributes (`strides`, `padding`).

        Returns:
            OpValue: Inferred output shape `(batch, *out_spatial, c_out)`.
        """
        strides = kwargs.get("strides", 1)
        padding = kwargs.get("padding", "VALID")

        batch = extract_shape(lhs)[0]
        c_out = extract_shape(rhs)[-1]

        spatial_in = extract_shape(lhs)[1:-1]
        spatial_k = extract_shape(rhs)[:-2]

        if isinstance(strides, int):
            strides = (strides,) * len(spatial_in)

        out_spatial = []
        for s_in, k, st in zip(spatial_in, spatial_k, coerce_shape(strides)):
            if padding == "VALID":
                s_out = (s_in - 1) * st + k
            else:
                s_out = s_in * st
            out_spatial.append(s_out)

        return (batch,) + tuple(out_spatial) + (c_out,)


@dispatch_eager("ConvGeneralDilated")
def conv_general_dilated(
    lhs: Tensor,
    rhs: Tensor,
    config: ConvConfig,
) -> OpValue:
    """Execute or trace general N-dimensional convolution with strides, padding, and dilations.

    Args:
        lhs (Tensor): Input activation tensor.
        rhs (Tensor): Convolution kernel tensor.
        config (ConvConfig): Convolution dimension numbers, strides, padding, and dilations.

    Returns:
        OpValue: Convolved output tensor.
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

    return _emit_linalg_node("ConvGeneralDilated", inputs, attributes, [out_shape], [lhs.dtype])


@dispatch_eager("ConvGeneralDilatedLocal")
def conv_general_dilated_local(
    lhs: Tensor,
    rhs: Tensor,
    config: ConvLocalHyperparams,
    **kwargs: OpValue,
) -> OpValue:
    """Execute or trace unshared-weights local N-dimensional dilated convolution.

    Args:
        lhs (Tensor): Input activation tensor.
        rhs (Tensor): Unshared local filter tensor.
        config (ConvLocalHyperparams): Local convolution hyperparameters.
        **kwargs (OpValue): Additional keyword attributes.

    Returns:
        OpValue: Locally convolved output tensor.
    """
    window_strides, padding, filter_shape = config.window_strides, config.padding, config.filter_shape
    inputs = [lhs, rhs]
    attributes = {
        "window_strides": window_strides,
        "padding": padding,
        "filter_shape": filter_shape,
        **kwargs,
    }

    out_shape = coerce_shape(ConvGeneralDilatedLocal().infer_shape(lhs, rhs, **attributes))
    return _emit_linalg_node("ConvGeneralDilatedLocal", inputs, attributes, [out_shape], [lhs.dtype])


@dispatch_eager("ConvGeneralDilatedPatches")
def conv_general_dilated_patches(
    lhs: Tensor,
    filter_shape: OpValue,
    window_strides: OpValue,
    padding: OpValue,
    **kwargs: OpValue,
) -> OpValue:
    """Extract dilated sliding-window spatial patches from `lhs`.

    Args:
        lhs (Tensor): Input activation tensor.
        filter_shape (OpValue): Spatial window dimensions.
        window_strides (OpValue): Spatial stride lengths.
        padding (OpValue): Padding specification.
        **kwargs (OpValue): Additional keyword attributes.

    Returns:
        OpValue: Extracted patches tensor.
    """
    inputs = [lhs]
    attributes = {
        "filter_shape": filter_shape,
        "window_strides": window_strides,
        "padding": padding,
        **kwargs,
    }

    out_shape = coerce_shape(ConvGeneralDilatedPatches().infer_shape(lhs, **attributes))
    return _emit_linalg_node("ConvGeneralDilatedPatches", inputs, attributes, [out_shape], [lhs.dtype])


@dispatch_eager("ConvWithGeneralPadding")
def conv_with_general_padding(
    lhs: Tensor,
    rhs: Tensor,
    window_strides: OpValue,
    padding: OpValue,
    **kwargs: OpValue,
) -> OpValue:
    """Execute or trace convolution with explicit asymmetric spatial padding.

    Args:
        lhs (Tensor): Input activation tensor.
        rhs (Tensor): Convolution kernel tensor.
        window_strides (OpValue): Spatial stride lengths.
        padding (OpValue): Explicit padding specification.
        **kwargs (OpValue): Additional keyword attributes.

    Returns:
        OpValue: Convolved output tensor.
    """
    inputs = [lhs, rhs]
    attributes = {"window_strides": window_strides, "padding": padding, **kwargs}

    out_shape = coerce_shape(ConvWithGeneralPadding().infer_shape(lhs, rhs, **attributes))
    return _emit_linalg_node("ConvWithGeneralPadding", inputs, attributes, [out_shape], [lhs.dtype])


__all__ = [
    "ConvGeneralDilated",
    "ConvGeneralDilatedLocal",
    "ConvGeneralDilatedPatches",
    "ConvLocalHyperparams",
    "ConvTranspose",
    "ConvTransposeShapeTuple",
    "ConvWithGeneralPadding",
    "Convolve",
    "conv_general_dilated",
    "conv_general_dilated_local",
    "conv_general_dilated_patches",
    "conv_with_general_padding",
]
