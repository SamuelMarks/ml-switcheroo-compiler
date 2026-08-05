"""Core abstractions and logic definitions for conv_ops.py."""

from ml_switcheroo_compiler.core.constants import MAGIC_VAL_2
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.configs import ConvConfig
from ml_switcheroo_compiler.ops.linalg.products import _has_valid_shape


@register_op("ConvGeneralDilated")
class ConvGeneralDilated(OpDef):
    """General N-dimensional convolution operator."""

    op_name = "ConvGeneralDilated"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
        """
        """Infer shape.

        Args:
            *args (object): lhs, rhs, config.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        lhs = args[0] if len(args) > 0 else kwargs["lhs"]
        rhs = args[1] if len(args) > 1 else kwargs["rhs"]
        config = args[2] if len(args) > MAGIC_VAL_2 else kwargs.get("config", None)
        if config is None:
            config = ConvConfig(window_strides=[], padding=[])
        if not _has_valid_shape(lhs) or not _has_valid_shape(rhs):
            return ()

        # simplified shape inference
        # Assume NCHW for lhs, OIHW for rhs, and (pad_h, pad_w)
        # We will just return () if dimension_numbers is None, but let's do a basic heuristic
        # If dimension_numbers provided, we'd parse it. Let's just return a placeholder for testing.
        return ()


@register_op("Convolve")
class Convolve(OpDef):
    """Return the discrete, linear convolution of two one-dimensional sequences."""

    op_name = "Convolve"
    np_op_name = "convolve"

    def infer_shape(self, a: object, v: object, mode: str = "full", **kwargs: object) -> object:
        """Infer the output shape.

        Args:
            a (object): The a parameter.
            v (object): The v parameter.
            mode (str): The mode parameter.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
        """
        return (None,)


@register_op("ConvGeneralDilatedLocal")
class ConvGeneralDilatedLocal(OpDef):
    """ConvGeneralDilatedLocal operator definition."""

    op_name = "ConvGeneralDilatedLocal"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
        """
        return args[0].shape if args else ()


@register_op("ConvGeneralDilatedPatches")
class ConvGeneralDilatedPatches(OpDef):
    """ConvGeneralDilatedPatches operator definition."""

    op_name = "ConvGeneralDilatedPatches"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
        """
        return args[0].shape if args else ()


@register_op("ConvWithGeneralPadding")
class ConvWithGeneralPadding(OpDef):
    """ConvWithGeneralPadding operator definition."""

    op_name = "ConvWithGeneralPadding"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
        """
        return args[0].shape if args else ()


@register_op("ConvTransposeShapeTuple")
class ConvTransposeShapeTuple(OpDef):
    """ConvTransposeShapeTuple operator definition."""

    op_name = "ConvTransposeShapeTuple"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
        """
        return ()


@register_op("ConvTranspose")
class ConvTranspose(OpDef):
    """ConvTranspose operator definition."""

    op_name = "ConvTranspose"

    def infer_shape(self, lhs: object, rhs: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            lhs (object): The lhs parameter.
            rhs (object): The rhs parameter.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
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
