"""Padding, image resizing, and shape metadata inspection operations."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.registry import coerce_int
from ml_switcheroo_compiler.core.shape import OpValue, extract_shape
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.base import OpDef, dispatch_eager, register_op
from ml_switcheroo_compiler.ops.shape.reshape import Resize
from ml_switcheroo_compiler.ops.shape.utils import _extract_shape_dtype


def _normalize_pad_width(pad_width: OpValue, ndim: int) -> tuple[OpValue, ...]:
    """Normalize a padding width specification across `ndim` tensor dimensions.

    Args:
        pad_width (OpValue): Scalar, pair, or per-axis sequence of padding widths.
        ndim (int): Rank of the target tensor.

    Returns:
        tuple[OpValue, ...]: Per-dimension `(before, after)` padding descriptors.
    """
    if isinstance(pad_width, int):
        return ((pad_width, pad_width),) * ndim
    if isinstance(pad_width, tuple) and len(pad_width) == 2 and isinstance(pad_width[0], int):
        return (pad_width,) * ndim
    return tuple(pad_width) if isinstance(pad_width, (list, tuple)) else ()


def _compute_pad_dim(dim: int, pw: OpValue) -> int:
    """Compute the padded size of a single dimension given its padding descriptor.

    Args:
        dim (int): Unpadded size of the dimension.
        pw (OpValue): Symmetric integer width or `(before, after)` padding pair.

    Returns:
        int: Resulting dimension size after padding.
    """
    if isinstance(pw, int):
        return dim + pw * 2
    if isinstance(pw, tuple) and len(pw) == 2:
        return dim + int(coerce_int(pw[0])) + int(coerce_int(pw[1]))
    return dim


@register_op("Pad")
class Pad(OpDef):
    """Schema definition and shape inference for array padding."""

    op_name = "Pad"

    def infer_shape(
        self,
        array: OpValue,
        pad_width: OpValue,
        mode: str = "constant",
        **kwargs: OpValue,
    ) -> OpValue:
        """Infer the output shape produced by padding `array` with `pad_width`.

        Args:
            array (OpValue): Input tensor or array-like operand to pad.
            pad_width (OpValue): Padding widths applied per dimension.
            mode (str): Boundary padding mode identifier.
            **kwargs (OpValue): Mode-specific keyword options such as `constant_values`.

        Returns:
            OpValue: Inferred padded output shape tuple.
        """
        del mode, kwargs
        in_shape = extract_shape(array)
        if not in_shape:
            return ()
        normalized_pw = _normalize_pad_width(pad_width, len(in_shape))
        out_shape = []
        for i, dim in enumerate(in_shape):
            if i < len(normalized_pw):
                out_shape.append(_compute_pad_dim(dim, normalized_pw[i]))
            else:
                out_shape.append(dim)
        return tuple(out_shape)


@dispatch_eager("Pad")
def pad(
    array: OpValue,
    pad_width: OpValue,
    mode: str = "constant",
    **kwargs: OpValue,
) -> OpValue:
    """Pad an array with specified per-axis widths and boundary handling mode.

    Args:
        array (OpValue): Input tensor or array-like operand to pad.
        pad_width (OpValue): Padding widths applied before and after each axis.
        mode (str): Padding mode (`'constant'`, `'reflect'`, `'edge'`, `'wrap'`, etc.).
        **kwargs (OpValue): Additional mode-specific attributes such as `constant_values`.

    Returns:
        OpValue: Padded tensor output.
    """
    import ml_switcheroo_compiler.ops.shape.pad_and_tile as _pt_mod

    op = Pad()
    out_shape = op.infer_shape(array, pad_width, mode, **kwargs)
    attributes = {"pad_width": pad_width, "mode": mode}
    attributes.update(kwargs)
    return _pt_mod._emit_shape_node("Pad", [array], attributes, out_shape, _extract_shape_dtype(array, None))


def pad_constant(
    array: OpValue,
    pad_width: OpValue,
    value: float = 0.0,
    **kwargs: OpValue,
) -> OpValue:
    """Pad an array using a constant fill value.

    Args:
        array (OpValue): Input tensor or array-like operand to pad.
        pad_width (OpValue): Padding widths applied before and after each axis.
        value (float): Constant scalar value inserted into padded regions.
        **kwargs (OpValue): Additional keyword options forwarded to `pad`.

    Returns:
        OpValue: Constant-padded tensor output.
    """
    import ml_switcheroo_compiler.ops.shape.pad_and_tile as _pt_mod

    return _pt_mod.pad(array, pad_width, mode="constant", constant_values=value, **kwargs)


def pad_reflect(array: OpValue, pad_width: OpValue, **kwargs: OpValue) -> OpValue:
    """Pad an array by reflecting values across boundary elements.

    Args:
        array (OpValue): Input tensor or array-like operand to pad.
        pad_width (OpValue): Padding widths applied before and after each axis.
        **kwargs (OpValue): Additional keyword options forwarded to `pad`.

    Returns:
        OpValue: Reflection-padded tensor output.
    """
    import ml_switcheroo_compiler.ops.shape.pad_and_tile as _pt_mod

    return _pt_mod.pad(array, pad_width, mode="reflect", **kwargs)


def pad_replicate(array: OpValue, pad_width: OpValue, **kwargs: OpValue) -> OpValue:
    """Pad an array by replicating edge values along boundaries.

    Args:
        array (OpValue): Input tensor or array-like operand to pad.
        pad_width (OpValue): Padding widths applied before and after each axis.
        **kwargs (OpValue): Additional keyword options forwarded to `pad`.

    Returns:
        OpValue: Edge-replicated padded tensor output.
    """
    import ml_switcheroo_compiler.ops.shape.pad_and_tile as _pt_mod

    return _pt_mod.pad(array, pad_width, mode="edge", **kwargs)


def pad_circular(array: OpValue, pad_width: OpValue, **kwargs: OpValue) -> OpValue:
    """Pad an array using circular wrap-around along each padded axis.

    Args:
        array (OpValue): Input tensor or array-like operand to pad.
        pad_width (OpValue): Padding widths applied before and after each axis.
        **kwargs (OpValue): Additional keyword options forwarded to `pad`.

    Returns:
        OpValue: Circularly wrapped padded tensor output.
    """
    import ml_switcheroo_compiler.ops.shape.pad_and_tile as _pt_mod

    return _pt_mod.pad(array, pad_width, mode="wrap", **kwargs)


@dispatch_eager("Resize")
def image_resize(image: Tensor, shape: tuple[int, int], method: str = "bilinear") -> OpValue:
    """Resize a spatial image tensor to `shape` using the given interpolation method.

    Args:
        image (Tensor): Input image tensor to resize.
        shape (tuple[int, int]): Target `(height, width)` spatial dimensions.
        method (str): Interpolation algorithm name (for example `'bilinear'` or `'nearest'`).

    Returns:
        OpValue: Resized image tensor.
    """
    import ml_switcheroo_compiler.ops.shape.pad_and_tile as _pt_mod

    op = Resize()
    out_shape = op.infer_shape(image, shape, method)
    return _pt_mod._emit_shape_node(
        "Resize",
        [image],
        {"shape": shape, "method": method},
        out_shape,
        image.dtype,
    )


@register_op("DynamicShape")
class DynamicShape(OpDef):
    """Schema definition returning a 1D tensor of runtime dimension sizes."""

    def infer_shape(self, x: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer the 1D shape vector length equal to the rank of `x`.

        Args:
            x (OpValue): Input tensor whose dynamic shape is queried.
            **kwargs (OpValue): Unused keyword operation attributes.

        Returns:
            OpValue: Single-element tuple containing the rank of `x`.
        """
        del kwargs
        return (len(extract_shape(x)),)


@register_op("Rank")
class Rank(OpDef):
    """Schema definition returning the scalar rank of an input tensor."""

    op_name = "Rank"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer the scalar output shape `()` for rank queries.

        Args:
            *args (OpValue): Input tensor operand positional arguments.
            **kwargs (OpValue): Unused keyword operation attributes.

        Returns:
            OpValue: Empty tuple representing a scalar rank result.
        """
        del args, kwargs
        return ()


@register_op("Size")
class Size(OpDef):
    """Schema definition returning the scalar element count of a tensor."""

    op_name = "Size"

    def infer_shape(self, a: OpValue, axis: OpValue = None, **kwargs: OpValue) -> OpValue:
        """Infer the scalar output shape `()` for element count queries.

        Args:
            a (OpValue): Input tensor whose element count is queried.
            axis (OpValue): Optional axis along which size is measured.
            **kwargs (OpValue): Unused keyword operation attributes.

        Returns:
            OpValue: Empty tuple representing a scalar size result.
        """
        del a, axis, kwargs
        return ()
