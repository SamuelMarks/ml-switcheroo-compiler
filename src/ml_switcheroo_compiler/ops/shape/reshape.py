# pylint: disable=duplicate-code

"""Define shape manipulation operations for the ML Switcheroo framework."""

from __future__ import annotations

import typing
from collections.abc import Sequence

from ml_switcheroo_compiler.backends.registry import coerce_int
from ml_switcheroo_compiler.core.constants import MIN_CONV_RANK
from ml_switcheroo_compiler.core.shape import OpValue, broadcast_shapes
from ml_switcheroo_compiler.ops.base import OpDef, register_op


@typing.runtime_checkable
class _ShapedOperandProtocol(typing.Protocol):
    """Protocol for tensor-like operands exposing a `shape` attribute."""

    shape: Sequence[int | None] | int | None


def _extract_valid_shape(x: OpValue) -> Sequence[int | None] | None:
    """Extract `.shape` from `x` as a sequence if non-None, otherwise return None.

    Args:
        x (OpValue): Input operand to check.

    Returns:
        Sequence[int | None] | None: Extracted shape sequence or None.
    """
    cand_x: OpValue | _ShapedOperandProtocol = x
    if isinstance(cand_x, _ShapedOperandProtocol) and cand_x.shape is not None:
        if isinstance(cand_x.shape, (tuple, list)):
            return cand_x.shape
        if isinstance(cand_x.shape, int):
            return (cand_x.shape,)
    return None


def _has_valid_shape(x: OpValue) -> bool:
    """Return True if `x` exposes a valid sequence `.shape` attribute.

    Args:
        x (OpValue): Input operand to check.

    Returns:
        bool: True if `x` has a non-None sequence `shape` attribute.
    """
    return _extract_valid_shape(x) is not None


def _iter_ints(val: OpValue) -> list[int]:
    """Convert an iterable or sequence of integers into a list of ints.

    Args:
        val (OpValue): Sequence or iterable of integer indices/dimensions.

    Returns:
        list[int]: Extracted integer list.
    """
    if isinstance(val, (tuple, list, Sequence)) and not isinstance(val, (str, bytes)):
        return [int(coerce_int(v)) for v in val]
    return [int(coerce_int(val))]


def _get_shape_list(x: OpValue) -> list[int]:
    """Help to convert shape to list.

    Args:
        x (OpValue): Input `x` argument (x).

    Returns:
        list: Computed `list` output.
    """
    valid_s = _extract_valid_shape(x)
    s: Sequence[int | None] | OpValue = valid_s if valid_s is not None else x
    if isinstance(s, (tuple, list)):
        return [coerce_int(d) for d in s]
    return [s] if isinstance(s, int) else []


def _normalize_axes(axes: OpValue, length: int) -> list[int]:
    """Help to normalize axes.

    Args:
        axes (OpValue): Input `axes` argument (axes).
        length (int): Input `length` argument (length).

    Returns:
        list: Computed `list` output.
    """
    a = [axes] if isinstance(axes, int) else (list(axes) if isinstance(axes, (list, tuple)) else [])
    return [coerce_int(x) + length if coerce_int(x) < 0 else coerce_int(x) for x in a]


def _resolve_reshape_minus_one(x_shape: Sequence[int | None], newshape: list[int]) -> list[int]:
    """Resolve the -1 dimension in a reshape operation.

    Args:
        x_shape (Sequence[int | None]): Input `x_shape` argument (x shape).
        newshape (list[int]): Input `newshape` argument (newshape).

    Returns:
        list: Computed `list` output.
    """
    import math

    known = math.prod(s for s in newshape if s != -1 and s is not None)
    total = math.prod(s for s in x_shape if s is not None)
    if total > 0 and known > 0:
        newshape[newshape.index(-1)] = total // known
    return newshape


@register_op("Reshape")
class Reshape(OpDef):
    """Provide an operator definition for reshaping a tensor to a new shape."""

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        x = args[0] if args else kwargs.get("a", kwargs.get("x"))
        newshape = args[1] if len(args) > 1 else kwargs["newshape"]

        x_shape = _extract_valid_shape(x)
        if x_shape is None or not isinstance(newshape, (tuple, list)):
            return newshape

        out_shape = [int(coerce_int(d)) for d in newshape]
        if -1 in out_shape:
            out_shape = _resolve_reshape_minus_one(x_shape, out_shape)

        return tuple(out_shape)


@register_op("Transpose")
class Transpose(OpDef):
    """Provide an operator definition for transposing the dimensions of a tensor."""

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer the output shape of the operation.

        Args:
            *args (OpValue): The first input tensor and the axes.
            **kwargs (OpValue): Additional keyword arguments.

        Returns:
            The computed shape or evaluation result.
        """
        x = args[0] if len(args) > 0 else kwargs["x"]
        axes = args[1] if len(args) > 1 else kwargs.get("axes", None)
        if isinstance(x, tuple) and axes is not None:
            return tuple(x[i] for i in _iter_ints(axes))
        return tuple(reversed(x)) if isinstance(x, tuple) else None

    def _format_args(self, x: str, axes: OpValue) -> str:
        """Compute the format args tensor operation.

        Args:
        x (str): Input `x` argument (x).
        axes (OpValue): Input `axes` argument (axes).

        Returns:
        str: Computed `str` output.
        """
        return f"{x}" if axes is None else f"{x}, {axes}"


@register_op("BroadcastTo")
class BroadcastTo(OpDef):
    """Provide an operator definition for broadcasting a tensor to a new shape."""

    def infer_shape(self, x: OpValue, shape: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer the output shape of the operation.

        Args:
            x (OpValue): Input `x` argument (x).
            shape (OpValue): Input `shape` argument (shape).
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.

        Raises:
            ValueError: If invalid inputs or unsupported runtime conditions trigger `ValueError`.
        """
        if not (isinstance(x, tuple) and isinstance(shape, tuple)):
            return shape

        try:
            broadcasted = broadcast_shapes(x, shape)
            if broadcasted != shape:
                raise ValueError(f"[broadcast_shapes] Shapes {x} and {shape} cannot be broadcast.")
        except ValueError as e:
            raise ValueError(f"[broadcast_shapes] Shapes {x} and {shape} cannot be broadcast.") from e

        return shape


@register_op("BroadcastInDim")
class BroadcastInDim(OpDef):
    """Provide an operator definition for broadcasting a tensor in a given set of dimensions."""

    op_name = "BroadcastInDim"

    def infer_shape(
        self,
        x: OpValue,
        shape: OpValue,
        broadcast_dimensions: OpValue,
        **kwargs: OpValue,
    ) -> OpValue:
        """Infer the output shape of the broadcasting operation.

        Args:
            x (OpValue): The input x tensor.
            shape (OpValue): The target shape.
            broadcast_dimensions (OpValue): The broadcast_dimensions parameter for the operation.
            **kwargs (OpValue): Additional keyword arguments.

        Returns: Tensor: The computed result.
        """
        return tuple(_iter_ints(shape))


@register_op("Resize")
class Resize(OpDef):
    """Provide an operator definition for resizing an image tensor to a target shape."""

    op_name = "Resize"

    def infer_shape(
        self,
        image: OpValue,
        shape: OpValue,
        method: OpValue = "bilinear",
        **kwargs: OpValue,
    ) -> OpValue:
        """Infer the output shape of the resizing operation.

        Args:
            image (OpValue): The image parameter for the operation.
            shape (OpValue): The target shape.
            method (OpValue): The method parameter for the operation.
            **kwargs (OpValue): Additional keyword arguments.

        Returns: Tensor: The computed result.
        """
        img_shape = _extract_valid_shape(image)
        if img_shape is None or not img_shape:
            return ()
        out_shape = [coerce_int(d) for d in img_shape]
        # assuming shape is (new_h, new_w) and image is either (..., H, W, C) or something.
        # Often it's (..., H, W, C) in Keras.
        if len(out_shape) >= MIN_CONV_RANK:
            shape_dims = _iter_ints(shape)
            out_shape[-3] = shape_dims[0]
            out_shape[-2] = shape_dims[1]
        return tuple(out_shape)


@register_op("Atleast1d")
class Atleast1d(OpDef):
    """Provide an operator definition for atleast_1d."""

    op_name = "Atleast1d"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        x = args[0] if len(args) > 0 else kwargs.get("a", kwargs.get("x"))
        if not _has_valid_shape(x):
            return (1,)
        shape = _get_shape_list(x)
        if len(shape) < 1:
            shape.insert(0, 1)
        return tuple(shape)


@register_op("Atleast2d")
class Atleast2d(OpDef):
    """Provide an operator definition for atleast_2d."""

    op_name = "Atleast2d"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        x = args[0] if len(args) > 0 else kwargs.get("a", kwargs.get("x"))
        if not _has_valid_shape(x):
            return (1, 1)
        shape = _get_shape_list(x)
        if len(shape) == 0:
            return (1, 1)
        if len(shape) == 1:
            return (1, shape[0])
        return tuple(shape)


@register_op("Atleast3d")
class Atleast3d(OpDef):
    """Provide an operator definition for atleast_3d."""

    op_name = "Atleast3d"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        x = args[0] if len(args) > 0 else kwargs.get("a", kwargs.get("x"))
        if not _has_valid_shape(x):
            return (1, 1, 1)
        shape = _get_shape_list(x)
        if len(shape) == 0:
            return (1, 1, 1)
        if len(shape) == 1:
            return (1, shape[0], 1)
        if len(shape) == 2:
            return (shape[0], shape[1], 1)
        return tuple(shape)


@register_op("ExpandDims")
class ExpandDims(OpDef):
    """Provide an operator definition for expanding the dimensions of a tensor."""

    op_name = "ExpandDims"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        x = args[0] if len(args) > 0 else kwargs.get("a", kwargs.get("x"))
        axis = args[1] if len(args) > 1 else kwargs.get("axis")
        if not _has_valid_shape(x):
            return (None,)
        shape = _get_shape_list(x)
        if axis is None:
            return tuple(shape)
        axis_idx = int(coerce_int(axis))
        if axis_idx < 0:
            axis_idx = axis_idx + len(shape) + 1
        shape.insert(axis_idx, 1)
        return tuple(shape)


@register_op("Block")
class Block(OpDef):
    """Block operator."""

    op_name = "Block"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        arrays = args[0] if args else kwargs.get("arrays")
        if not isinstance(arrays, (list, tuple)):
            return ()
        return ()


@register_op("C")
class C(OpDef):
    """C operation for concatenation shorthand."""

    op_name = "C"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        return ()


@register_op("Collapse")
class Collapse(OpDef):
    """Collapse operation for shape manipulation."""

    op_name = "Collapse"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        return ()


@register_op("Delete")
class Delete(OpDef):
    """Delete operator."""

    op_name = "Delete"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        arr = args[0] if args else kwargs.get("arr")
        arr_shape = _extract_valid_shape(arr)
        if arr_shape is None:
            return None
        return tuple(arr_shape)  # Approximate


@register_op("DiagIndices")
class DiagIndices(OpDef):
    """DiagIndices operator."""

    op_name = "DiagIndices"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        n = args[0] if args else kwargs.get("n")
        ndim = args[1] if len(args) > 1 else kwargs.get("ndim", 2)
        return tuple([(n,)] * int(coerce_int(ndim)))


@register_op("DiagIndicesFrom")
class DiagIndicesFrom(OpDef):
    """DiagIndicesFrom operator."""

    op_name = "DiagIndicesFrom"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        arr = args[0] if args else kwargs.get("arr")
        arr_shape = _extract_valid_shape(arr)
        if arr_shape is None:
            return None
        return tuple([(arr_shape[0],)] * len(arr_shape))


@register_op("Diagflat")
class Diagflat(OpDef):
    """Diagflat operator."""

    op_name = "Diagflat"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        v = args[0] if args else kwargs.get("v")
        v_shape = _extract_valid_shape(v)
        if v_shape is None:
            return None
        import math

        size = math.prod(coerce_int(d) for d in v_shape)
        k = args[1] if len(args) > 1 else kwargs.get("k", 0)
        s = size + abs(int(coerce_int(k)))
        return (s, s)


@register_op("FillDiagonal")
class FillDiagonal(OpDef):
    """FillDiagonal operator."""

    op_name = "FillDiagonal"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        a = args[0] if args else kwargs.get("a")
        a_shape = _extract_valid_shape(a)
        if a_shape is None:
            return None
        return tuple(a_shape)


@register_op("Insert")
class Insert(OpDef):
    """Insert operator."""

    op_name = "Insert"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        arr = args[0] if args else kwargs.get("arr")
        arr_shape = _extract_valid_shape(arr)
        if arr_shape is None:
            return None
        return tuple(arr_shape)  # Approximate


@register_op("Moveaxis")
class Moveaxis(OpDef):
    """Provide an operator definition for moving axes of a tensor."""

    op_name = "Moveaxis"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        x = args[0] if args else kwargs.get("a", kwargs.get("x"))
        if not _has_valid_shape(x):
            return None
        shape = _get_shape_list(x)
        source = args[1] if len(args) > 1 else kwargs.get("source")
        destination = args[2] if len(args) > 2 else kwargs.get("destination")
        return self._calc_shape(shape, source, destination)

    def _calc_shape(self, shape: list[int], source: OpValue, destination: OpValue) -> OpValue:
        """Calculate the output shape for a moveaxis operation.

        Args:
            shape (list[int]): The input shape.
            source (OpValue): The source axes.
            destination (OpValue): The destination axes.

        Returns:
            tuple: The calculated shape.
        """
        length = len(shape)
        sl = _normalize_axes(source, length)
        dl = _normalize_axes(destination, length)
        order = [i for i in range(length) if i not in sl]
        for dest, src in sorted(zip(dl, sl)):
            order.insert(dest, src)
        return tuple(shape[i] for i in order)


@register_op("Permute")
class Permute(OpDef):
    """Provide an operator definition for permuting the dimensions of a tensor."""

    op_name = "Permute"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        x = args[0] if args else kwargs.get("a", kwargs.get("x"))
        if not isinstance(x, (tuple, list)) and not _has_valid_shape(x):
            return None
        shape = _get_shape_list(x)
        dims = args[1] if len(args) > 1 else kwargs.get("dims")
        if dims is None:
            return tuple(shape[::-1])
        return tuple(shape[i] for i in _iter_ints(dims))


@register_op("Roll")
class Roll(OpDef):
    """Provide an operator definition for rolling array elements along a given axis."""

    op_name = "Roll"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        x = args[0] if args else kwargs.get("a", kwargs.get("x"))
        if not isinstance(x, (tuple, list)) and not _has_valid_shape(x):
            return None
        return tuple(_get_shape_list(x))


@register_op("Squeeze")
class Squeeze(OpDef):
    """Provide an operator definition for squeezing dimensions of a tensor."""

    op_name = "Squeeze"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        x = args[0] if args else kwargs.get("a", kwargs.get("x"))
        if not isinstance(x, (tuple, list)) and not _has_valid_shape(x):
            return None
        shape = _get_shape_list(x)
        axis = args[1] if len(args) > 1 else kwargs.get("axis")
        return self._calc_shape(shape, axis)

    def _calc_shape(self, shape: list[int], axis: OpValue) -> OpValue:
        """Calculate the output shape for a squeeze operation.

        Args:
            shape (list[int]): The input shape.
            axis (OpValue): The axis or axes to squeeze.

        Returns:
            tuple: The calculated shape.
        """
        if axis is None:
            return tuple(s for s in shape if s != 1)
        n = {a + len(shape) if a < 0 else a for a in _iter_ints(axis)}
        return tuple(shape[i] for i in range(len(shape)) if i not in n)


@register_op("Swapaxes")
class Swapaxes(OpDef):
    """Provide an operator definition for swapping two axes of an array."""

    op_name = "Swapaxes"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        x = args[0] if args else kwargs.get("a", kwargs.get("x"))
        if not isinstance(x, (tuple, list)) and not _has_valid_shape(x):
            return None
        shape = _get_shape_list(x)
        axis1 = int(coerce_int(args[1] if len(args) > 1 else kwargs.get("axis1", 0)))
        axis2 = int(coerce_int(args[2] if len(args) > 2 else kwargs.get("axis2", 0)))
        a1 = axis1 + len(shape) if axis1 < 0 else axis1
        a2 = axis2 + len(shape) if axis2 < 0 else axis2
        shape[a1], shape[a2] = shape[a2], shape[a1]
        return tuple(shape)


@register_op("Flip")
class Flip(OpDef):
    """Provide an operator definition for flipping an array."""

    op_name = "Flip"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        x = args[0] if args else kwargs.get("m", kwargs.get("x"))
        if not isinstance(x, (tuple, list)) and not _has_valid_shape(x):
            return None
        return tuple(_get_shape_list(x))


@register_op("Fliplr")
class Fliplr(OpDef):
    """Provide an operator definition for flipping an array left/right."""

    op_name = "Fliplr"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        x = args[0] if args else kwargs.get("m", kwargs.get("x"))
        if not isinstance(x, (tuple, list)) and not _has_valid_shape(x):
            return None
        return tuple(_get_shape_list(x))


@register_op("Flipud")
class Flipud(OpDef):
    """Provide an operator definition for flipping an array up/down."""

    op_name = "Flipud"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        x = args[0] if args else kwargs.get("m", kwargs.get("x"))
        if not isinstance(x, (tuple, list)) and not _has_valid_shape(x):
            return None
        return tuple(_get_shape_list(x))
