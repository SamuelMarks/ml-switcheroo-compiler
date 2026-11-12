# pylint: disable=duplicate-code

"""Define shape manipulation operations for the ML Switcheroo framework."""

from __future__ import annotations

import typing
from collections.abc import Sequence

from ml_switcheroo_compiler.backends.registry import EagerTensorProtocol, ensure_int
from ml_switcheroo_compiler.core.constants import MIN_CONV_RANK
from ml_switcheroo_compiler.core.shape import TensorOrScalar, broadcast_shapes
from ml_switcheroo_compiler.ops.base import OpDef, register_op


@typing.runtime_checkable
class _ShapedOperandProtocol(typing.Protocol):
    """Protocol for tensor-like operands exposing an integer `shape` sequence."""

    shape: Sequence[int]


def _has_valid_shape(x: TensorOrScalar) -> typing.TypeGuard[_ShapedOperandProtocol]:
    """Return True if `x` implements `_ShapedOperandProtocol` with a non-None `shape`.

    Args:
        x (TensorOrScalar): Input operand to check.

    Returns:
        bool: True if `x` has a non-None `shape` attribute.
    """
    return isinstance(x, _ShapedOperandProtocol) and x.shape is not None


def _iter_ints(val: TensorOrScalar) -> list[int]:
    """Convert an iterable or sequence of integers into a list of ints.

    Args:
        val (TensorOrScalar): Sequence or iterable of integer indices/dimensions.

    Returns:
        list[int]: Extracted integer list.
    """
    if isinstance(val, (tuple, list, Sequence)) and not isinstance(val, (str, bytes)):
        return [int(ensure_int(v)) for v in val]
    return [int(ensure_int(val))]


def _get_shape_list(x: TensorOrScalar) -> list[int]:
    """Help to convert shape to list.

    Args:
        x (TensorOrScalar): Input `x` argument (x).

    Returns:
        list: Computed `list` output.
    """
    s: Sequence[int] | TensorOrScalar = x.shape if _has_valid_shape(x) else x
    if isinstance(s, (tuple, list)):
        return [ensure_int(d) for d in s]
    return [s] if isinstance(s, int) else []


def _normalize_axes(axes: TensorOrScalar, length: int) -> list[int]:
    """Help to normalize axes.

    Args:
        axes (TensorOrScalar): Input `axes` argument (axes).
        length (int): Input `length` argument (length).

    Returns:
        list: Computed `list` output.
    """
    a = [axes] if isinstance(axes, int) else (list(axes) if isinstance(axes, (list, tuple)) else [])
    return [ensure_int(x) + length if ensure_int(x) < 0 else ensure_int(x) for x in a]


def _resolve_reshape_minus_one(x_shape: Sequence[int | None], newshape: list[int]) -> list[int]:
    """Resolve the -1 dimension in a reshape operation.

    Args:
        x_shape (tuple): Input `x_shape` argument (x shape).
        newshape (list): Input `newshape` argument (newshape).

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

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
        *args (TensorOrScalar): Positional args.
        **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        x = args[0] if args else kwargs.get("a", kwargs.get("x"))
        newshape = args[1] if len(args) > 1 else kwargs["newshape"]

        if not _has_valid_shape(x) or not isinstance(newshape, (tuple, list)):
            return newshape

        out_shape = [int(ensure_int(d)) for d in newshape]
        if -1 in out_shape:
            out_shape = _resolve_reshape_minus_one(x.shape, out_shape)

        return tuple(out_shape)


@register_op("Transpose")
class Transpose(OpDef):
    """Provide an operator definition for transposing the dimensions of a tensor."""

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer the output shape of the operation.

        Args:
            *args (TensorOrScalar): The first input tensor and the axes.
            **kwargs (TensorOrScalar): Additional keyword arguments.

        Returns:
            The computed shape or evaluation result.
        """
        x = args[0] if len(args) > 0 else kwargs["x"]
        axes = args[1] if len(args) > 1 else kwargs.get("axes", None)
        if isinstance(x, tuple) and axes is not None:
            return tuple(x[i] for i in _iter_ints(axes))
        return tuple(reversed(x)) if isinstance(x, tuple) else None

    def _format_args(self, x: str, axes: TensorOrScalar) -> str:
        """Compute the format args tensor operation.

        Args:
        x (str): Input `x` argument (x).
        axes (TensorOrScalar): Input `axes` argument (axes).

        Returns:
        str: Computed `str` output.
        """
        return f"{x}" if axes is None else f"{x}, {axes}"


@register_op("BroadcastTo")
class BroadcastTo(OpDef):
    """Provide an operator definition for broadcasting a tensor to a new shape."""

    def infer_shape(self, x: TensorOrScalar, shape: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer the output shape of the operation.

        Args:
            x (EagerTensorProtocol): Input `x` argument (x).
            shape (EagerTensorProtocol): Input `shape` argument (shape).
            **kwargs (TensorOrScalar): Keyword args.

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
        x: TensorOrScalar,
        shape: TensorOrScalar,
        broadcast_dimensions: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> TensorOrScalar:
        """Infer the output shape of the broadcasting operation.

        Args:
            x (EagerTensorProtocol): The input x tensor.
            shape (EagerTensorProtocol): The target shape.
            broadcast_dimensions (EagerTensorProtocol): The broadcast_dimensions parameter for the operation.
            **kwargs (TensorOrScalar): Additional keyword arguments.

        Returns: Tensor: The computed result.
        """
        return tuple(_iter_ints(shape))


@register_op("Resize")
class Resize(OpDef):
    """Provide an operator definition for resizing an image tensor to a target shape."""

    op_name = "Resize"

    def infer_shape(
        self,
        image: EagerTensorProtocol,
        shape: EagerTensorProtocol,
        method: TensorOrScalar = "bilinear",
        **kwargs: TensorOrScalar,
    ) -> TensorOrScalar:
        """Infer the output shape of the resizing operation.

        Args:
            image (EagerTensorProtocol): The image parameter for the operation.
            shape (EagerTensorProtocol): The target shape.
            method (TensorOrScalar): The method parameter for the operation.
            **kwargs (TensorOrScalar): Additional keyword arguments.

        Returns: Tensor: The computed result.
        """
        if not _has_valid_shape(image) or not image.shape:
            return ()
        out_shape = list(image.shape)
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

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            *args (TensorOrScalar): Positional args.
            **kwargs (TensorOrScalar): Keyword args.

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

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            *args (TensorOrScalar): Positional args.
            **kwargs (TensorOrScalar): Keyword args.

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

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            *args (TensorOrScalar): Positional args.
            **kwargs (TensorOrScalar): Keyword args.

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

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            *args (TensorOrScalar): Positional args.
            **kwargs (TensorOrScalar): Keyword args.

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
        axis_idx = int(ensure_int(axis))
        if axis_idx < 0:
            axis_idx = axis_idx + len(shape) + 1
        shape.insert(axis_idx, 1)
        return tuple(shape)


@register_op("Block")
class Block(OpDef):
    """Block operator."""

    op_name = "Block"

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            *args (TensorOrScalar): Positional args.
            **kwargs (TensorOrScalar): Keyword args.

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

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            *args (TensorOrScalar): Positional args.
            **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        return ()


@register_op("Collapse")
class Collapse(OpDef):
    """Collapse operation for shape manipulation."""

    op_name = "Collapse"

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            *args (TensorOrScalar): Positional args.
            **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        return ()


@register_op("Delete")
class Delete(OpDef):
    """Delete operator."""

    op_name = "Delete"

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            *args (TensorOrScalar): Positional args.
            **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        arr = args[0] if args else kwargs.get("arr")
        if not _has_valid_shape(arr):
            return None
        return tuple(arr.shape)  # Approximate


@register_op("DiagIndices")
class DiagIndices(OpDef):
    """DiagIndices operator."""

    op_name = "DiagIndices"

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            *args (TensorOrScalar): Positional args.
            **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        n = args[0] if args else kwargs.get("n")
        ndim = args[1] if len(args) > 1 else kwargs.get("ndim", 2)
        return tuple([(n,)] * int(ensure_int(ndim)))


@register_op("DiagIndicesFrom")
class DiagIndicesFrom(OpDef):
    """DiagIndicesFrom operator."""

    op_name = "DiagIndicesFrom"

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            *args (TensorOrScalar): Positional args.
            **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        arr = args[0] if args else kwargs.get("arr")
        if not _has_valid_shape(arr):
            return None
        return tuple([(arr.shape[0],)] * len(arr.shape))


@register_op("Diagflat")
class Diagflat(OpDef):
    """Diagflat operator."""

    op_name = "Diagflat"

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            *args (TensorOrScalar): Positional args.
            **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        v = args[0] if args else kwargs.get("v")
        if not _has_valid_shape(v):
            return None
        import math

        size = math.prod(v.shape)
        k = args[1] if len(args) > 1 else kwargs.get("k", 0)
        s = size + abs(int(ensure_int(k)))
        return (s, s)


@register_op("FillDiagonal")
class FillDiagonal(OpDef):
    """FillDiagonal operator."""

    op_name = "FillDiagonal"

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            *args (TensorOrScalar): Positional args.
            **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        a = args[0] if args else kwargs.get("a")
        if not _has_valid_shape(a):
            return None
        return tuple(a.shape)


@register_op("Insert")
class Insert(OpDef):
    """Insert operator."""

    op_name = "Insert"

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            *args (TensorOrScalar): Positional args.
            **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        arr = args[0] if args else kwargs.get("arr")
        if not _has_valid_shape(arr):
            return None
        return tuple(arr.shape)  # Approximate


@register_op("Moveaxis")
class Moveaxis(OpDef):
    """Provide an operator definition for moving axes of a tensor."""

    op_name = "Moveaxis"

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            *args (TensorOrScalar): Positional args.
            **kwargs (TensorOrScalar): Keyword args.

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

    def _calc_shape(self, shape: list[int], source: TensorOrScalar, destination: TensorOrScalar) -> TensorOrScalar:
        """Calculate the output shape for a moveaxis operation.

        Args:
            shape (list[int]): The input shape.
            source (TensorOrScalar): The source axes.
            destination (TensorOrScalar): The destination axes.

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

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            *args (TensorOrScalar): Positional args.
            **kwargs (TensorOrScalar): Keyword args.

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

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            *args (TensorOrScalar): Positional args.
            **kwargs (TensorOrScalar): Keyword args.

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

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            *args (TensorOrScalar): Positional args.
            **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        x = args[0] if args else kwargs.get("a", kwargs.get("x"))
        if not isinstance(x, (tuple, list)) and not _has_valid_shape(x):
            return None
        shape = _get_shape_list(x)
        axis = args[1] if len(args) > 1 else kwargs.get("axis")
        return self._calc_shape(shape, axis)

    def _calc_shape(self, shape: list[int], axis: TensorOrScalar) -> TensorOrScalar:
        """Calculate the output shape for a squeeze operation.

        Args:
            shape (list[int]): The input shape.
            axis (TensorOrScalar): The axis or axes to squeeze.

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

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            *args (TensorOrScalar): Positional args.
            **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        x = args[0] if args else kwargs.get("a", kwargs.get("x"))
        if not isinstance(x, (tuple, list)) and not _has_valid_shape(x):
            return None
        shape = _get_shape_list(x)
        axis1 = int(ensure_int(args[1] if len(args) > 1 else kwargs.get("axis1", 0)))
        axis2 = int(ensure_int(args[2] if len(args) > 2 else kwargs.get("axis2", 0)))
        a1 = axis1 + len(shape) if axis1 < 0 else axis1
        a2 = axis2 + len(shape) if axis2 < 0 else axis2
        shape[a1], shape[a2] = shape[a2], shape[a1]
        return tuple(shape)


@register_op("Flip")
class Flip(OpDef):
    """Provide an operator definition for flipping an array."""

    op_name = "Flip"

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            *args (TensorOrScalar): Positional args.
            **kwargs (TensorOrScalar): Keyword args.

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

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            *args (TensorOrScalar): Positional args.
            **kwargs (TensorOrScalar): Keyword args.

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

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            *args (TensorOrScalar): Positional args.
            **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        x = args[0] if args else kwargs.get("m", kwargs.get("x"))
        if not isinstance(x, (tuple, list)) and not _has_valid_shape(x):
            return None
        return tuple(_get_shape_list(x))
