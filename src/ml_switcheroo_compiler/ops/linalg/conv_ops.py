# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Core abstractions and logic definitions for conv_ops.py."""

from ml_switcheroo_compiler.core.constants import MAGIC_VAL_2
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.configs import ConvConfig
from ml_switcheroo_compiler.ops.linalg.products import _has_valid_shape


@register_op("ConvGeneralDilated")
class ConvGeneralDilated(OpDef):
    """General N-dimensional convolution operator."""

    op_name = "ConvGeneralDilated"

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        """Infer shape.

        Args:
            *args (Any): lhs, rhs, config.
            **kwargs: Additional keyword arguments.

        Returns: Tensor: The evaluated output resulting from this operation.
        """
        lhs = args[0] if len(args) > 0 else kwargs.get("lhs")
        rhs = args[1] if len(args) > 1 else kwargs.get("rhs")
        config = args[2] if len(args) > MAGIC_VAL_2 else kwargs.get("config", None)
        if config is None:
            config = ConvConfig(window_strides=[], padding=[])
        if not _has_valid_shape(lhs) or not _has_valid_shape(rhs):
            return ()

        lhs_shape = getattr(lhs, "shape", ())
        rhs_shape = getattr(rhs, "shape", ())
        if not lhs_shape or not rhs_shape or len(lhs_shape) < 3 or len(rhs_shape) < 3:
            return ()

        batch = lhs_shape[0]
        out_channels = rhs_shape[0]
        spatial_in = lhs_shape[2:]
        spatial_k = rhs_shape[2:]
        strides = getattr(config, "window_strides", []) or [1] * len(spatial_in)
        padding = getattr(config, "padding", []) or [(0, 0)] * len(spatial_in)
        lhs_dilation = getattr(config, "lhs_dilation", []) or [1] * len(spatial_in)
        rhs_dilation = getattr(config, "rhs_dilation", []) or [1] * len(spatial_in)

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
    """Return the discrete, linear convolution of two one-dimensional sequences."""

    op_name = "Convolve"
    np_op_name = "convolve"

    def infer_shape(self, a, v, mode: str = "full", **kwargs):
        """Infer the output shape for 1D discrete linear convolution.

        Args:
            a (Any): First input array.
            v (Any): Second input array.
            mode (str): Convolution mode ('full', 'same', or 'valid').
            **kwargs (Any): Additional keyword arguments.

        Returns:
            tuple[int, ...]: Inferred output shape.
        """
        if not _has_valid_shape(a) or not _has_valid_shape(v):
            return ()
        a_shape = getattr(a, "shape", ())
        v_shape = getattr(v, "shape", ())
        na = a_shape[0] if a_shape else 1
        nv = v_shape[0] if v_shape else 1
        if mode == "full":
            return (int(na) + int(nv) - 1,)
        elif mode == "same":
            return (max(int(na), int(nv)),)
        elif mode == "valid":
            return (max(int(na), int(nv)) - min(int(na), int(nv)) + 1,)
        return (int(na) + int(nv) - 1,)


@register_op("ConvGeneralDilatedLocal")
class ConvGeneralDilatedLocal(OpDef):
    """ConvGeneralDilatedLocal operator definition."""

    op_name = "ConvGeneralDilatedLocal"

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return args[0].shape if args else ()


@register_op("ConvGeneralDilatedPatches")
class ConvGeneralDilatedPatches(OpDef):
    """ConvGeneralDilatedPatches operator definition."""

    op_name = "ConvGeneralDilatedPatches"

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return args[0].shape if args else ()


@register_op("ConvWithGeneralPadding")
class ConvWithGeneralPadding(OpDef):
    """ConvWithGeneralPadding operator definition."""

    op_name = "ConvWithGeneralPadding"

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return args[0].shape if args else ()


@register_op("ConvTransposeShapeTuple")
class ConvTransposeShapeTuple(OpDef):
    """ConvTransposeShapeTuple operator definition."""

    op_name = "ConvTransposeShapeTuple"

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return ()


@register_op("ConvTranspose")
class ConvTranspose(OpDef):
    """ConvTranspose operator definition."""

    op_name = "ConvTranspose"

    def infer_shape(self, lhs, rhs, **kwargs):
        """Infer shape.

        Args:
            lhs (Any): The lhs parameter.
            rhs (Any): The rhs parameter.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        # Assuming NHWC and HWIO or NWC and WIO
        strides = kwargs.get("strides", 1)
        padding = kwargs.get("padding", "VALID")

        batch = lhs.shape[0]
        c_out = rhs.shape[-1]

        spatial_in = lhs.shape[1:-1]
        spatial_k = rhs.shape[:-2]

        if isinstance(strides, int):
            strides = (strides,) * len(spatial_in)

        out_spatial = []
        for s_in, k, st in zip(spatial_in, spatial_k, strides):
            if padding == "VALID":
                s_out = (s_in - 1) * st + k
            else:  # SAME
                s_out = s_in * st
            out_spatial.append(s_out)

        return (batch,) + tuple(out_spatial) + (c_out,)
