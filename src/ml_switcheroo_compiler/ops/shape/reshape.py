# pylint: disable=duplicate-code

"""Define shape manipulation operations for the ML Switcheroo framework."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Callable, cast

from ml_switcheroo_compiler.backends.registry import EagerTensorProtocol
from ml_switcheroo_compiler.core.constants import MIN_CONV_RANK
from ml_switcheroo_compiler.core.shape import TensorOrScalar, broadcast_shapes
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.base import OpDef, register_op


def _get_shape_list(x: Tensor | tuple[int, ...] | list[int] | int | None) -> list[int]:
    """Help to convert shape to list.

    Args:
        x (Tensor | tuple[int, ...] | list[int] | int | None): The x parameter.

    Returns:
        list: Result.
    """
    if isinstance(x, (tuple, list)):
        return list(x)
    if not hasattr(x, "shape") or getattr(x, "shape") is None:
        return []
    s = getattr(x, "shape")
    if isinstance(s, tuple):
        return list(s)
    if isinstance(s, int):
        return [s]
    if isinstance(s, list):
        return list(s)
    return []


def _normalize_axes(axes: int | Sequence[int], length: int) -> list[int]:
    """Help to normalize axes.

    Args:
        axes (int | Sequence[int]): The axes parameter.
        length (int): The length parameter.

    Returns:
        list: Result.
    """
    a = [axes] if isinstance(axes, int) else axes
    return [x + length if x < 0 else x for x in a]


def _resolve_reshape_minus_one(x_shape: Sequence[int | None], newshape: list[int]) -> list[int]:
    """Resolve the -1 dimension in a reshape operation.

    Args:
        x_shape (tuple): The x_shape parameter.
        newshape (list): The newshape parameter.

    Returns:
        list: Result.
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
            tuple[int, ...]: Result.
        """
        x = args[0] if args else kwargs.get("a", kwargs.get("x"))
        newshape = args[1] if len(args) > 1 else kwargs["newshape"]

        if not hasattr(x, "shape") or not isinstance(newshape, (tuple, list)):
            return newshape

        out_shape = list(newshape)
        if -1 in out_shape:
            out_shape = _resolve_reshape_minus_one(getattr(x, "shape"), out_shape)

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
            return tuple(x[i] for i in getattr(axes, "__iter__")())
        return tuple(reversed(x)) if isinstance(x, tuple) else None

    def _format_args(self, x: str, axes: TensorOrScalar) -> str:
        """Compute the format args tensor operation.

        Args:
        x (str): The x parameter.
        axes (TensorOrScalar): The axes parameter.

        Returns:
        str: Result.
        """
        return f"{x}" if axes is None else f"{x}, {axes}"


@register_op("BroadcastTo")
class BroadcastTo(OpDef):
    """Provide an operator definition for broadcasting a tensor to a new shape."""

    def infer_shape(self, x: EagerTensorProtocol, shape: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer the output shape of the operation.

        Args:
            x (EagerTensorProtocol): The x parameter.
            shape (EagerTensorProtocol): The shape parameter.
            **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple[int, ...]: Result.

        Raises:
            ValueError: An exception.
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
        x: EagerTensorProtocol,
        shape: EagerTensorProtocol,
        broadcast_dimensions: EagerTensorProtocol,
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
        return tuple(shape)


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
        if not hasattr(image, "shape") or not image.shape:
            return ()
        out_shape = list(image.shape)
        # assuming shape is (new_h, new_w) and image is either (..., H, W, C) or something.
        # Often it's (..., H, W, C) in Keras.
        if len(out_shape) >= MIN_CONV_RANK:
            cast(EagerTensorProtocol, out_shape)[-3] = getattr(shape, "__getitem__")(0)
            cast(EagerTensorProtocol, out_shape)[-2] = getattr(shape, "__getitem__")(1)
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
            tuple[int, ...]: Result.
        """
        x = args[0] if len(args) > 0 else kwargs.get("a", kwargs.get("x"))
        if not hasattr(x, "shape") or getattr(x, "shape") is None:
            return (1,)
        shape = cast("Callable[..., list[int]]", _get_shape_list)(x)
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
            tuple[int, ...]: Result.
        """
        x = args[0] if len(args) > 0 else kwargs.get("a", kwargs.get("x"))
        if not hasattr(x, "shape") or getattr(x, "shape") is None:
            return (1, 1)
        shape = cast("Callable[..., list[int]]", _get_shape_list)(x)
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
            tuple[int, ...]: Result.
        """
        x = args[0] if len(args) > 0 else kwargs.get("a", kwargs.get("x"))
        if not hasattr(x, "shape") or getattr(x, "shape") is None:
            return (1, 1, 1)
        shape = cast("Callable[..., list[int]]", _get_shape_list)(x)
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
            tuple[int, ...]: Result.
        """
        x = args[0] if len(args) > 0 else kwargs.get("a", kwargs.get("x"))
        axis = args[1] if len(args) > 1 else kwargs.get("axis")
        if not hasattr(x, "shape") or getattr(x, "shape") is None:
            return (None,)
        shape = cast("Callable[..., list[int]]", _get_shape_list)(x)
        if axis is None:
            return tuple(shape)
        if int(cast(int, axis)) < 0:
            axis = cast(EagerTensorProtocol, axis) + cast(EagerTensorProtocol, len(shape)) + 1
        getattr(shape, "insert")(axis, 1)
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
            tuple[int, ...]: Result.
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
            tuple[int, ...]: Result.
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
            tuple[int, ...]: Result.
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
            tuple[int, ...]: Result.
        """
        arr = args[0] if args else kwargs.get("arr")
        if arr is None or not hasattr(arr, "shape"):
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
            tuple[int, ...]: Result.
        """
        n = args[0] if args else kwargs.get("n")
        ndim = args[1] if len(args) > 1 else kwargs.get("ndim", 2)
        return tuple([(n,)] * int(cast(int, ndim)))


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
            tuple[int, ...]: Result.
        """
        arr = args[0] if args else kwargs.get("arr")
        if arr is None or not hasattr(arr, "shape"):
            return None
        return tuple([(getattr(arr.shape, "__getitem__")(0),)] * len(arr.shape))


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
            tuple[int, ...]: Result.
        """
        v = args[0] if args else kwargs.get("v")
        if not hasattr(v, "shape"):
            return None
        import math

        size = math.prod(getattr(v, "shape"))
        k = args[1] if len(args) > 1 else kwargs.get("k", 0)
        s = size + cast("Callable[..., EagerTensorProtocol]", abs)(k)
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
            tuple[int, ...]: Result.
        """
        a = args[0] if args else kwargs.get("a")
        if not hasattr(a, "shape"):
            return None
        return tuple(getattr(a, "shape"))


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
            tuple[int, ...]: Result.
        """
        arr = args[0] if args else kwargs.get("arr")
        if arr is None or not hasattr(arr, "shape"):
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
            tuple[int, ...]: Result.
        """
        x = args[0] if args else kwargs.get("a", kwargs.get("x"))
        if not hasattr(x, "shape") or getattr(x, "shape") is None:
            return None
        shape = cast("Callable[..., list[int]]", _get_shape_list)(x)
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
        sl = cast("Callable[..., list[int]]", _normalize_axes)(source, length)
        dl = cast("Callable[..., list[int]]", _normalize_axes)(destination, length)
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
            tuple[int, ...]: Result.
        """
        x = args[0] if args else kwargs.get("a", kwargs.get("x"))
        if not isinstance(x, (tuple, list)) and not hasattr(x, "shape"):
            return None
        shape = cast("Callable[..., list[int]]", _get_shape_list)(x)
        dims = args[1] if len(args) > 1 else kwargs.get("dims")
        if dims is None:
            return tuple(shape[::-1])
        return tuple(shape[i] for i in getattr(dims, "__iter__")())


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
            tuple[int, ...]: Result.
        """
        x = args[0] if args else kwargs.get("a", kwargs.get("x"))
        if not isinstance(x, (tuple, list)) and not hasattr(x, "shape"):
            return None
        return tuple(cast("Callable[..., list[int]]", _get_shape_list)(x))


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
            tuple[int, ...]: Result.
        """
        x = args[0] if args else kwargs.get("a", kwargs.get("x"))
        if not isinstance(x, (tuple, list)) and not hasattr(x, "shape"):
            return None
        shape = cast("Callable[..., list[int]]", _get_shape_list)(x)
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
        axes = [axis] if isinstance(axis, int) else axis
        n = {a + len(shape) if a < 0 else a for a in getattr(axes, "__iter__")()}
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
            tuple[int, ...]: Result.
        """
        x = args[0] if args else kwargs.get("a", kwargs.get("x"))
        if not isinstance(x, (tuple, list)) and not hasattr(x, "shape"):
            return None
        shape = cast("Callable[..., list[int]]", _get_shape_list)(x)
        axis1 = int(cast(int, args[1] if len(args) > 1 else kwargs.get("axis1", 0)))
        axis2 = int(cast(int, args[2] if len(args) > 2 else kwargs.get("axis2", 0)))
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
            tuple[int, ...]: Result.
        """
        x = args[0] if args else kwargs.get("m", kwargs.get("x"))
        if not isinstance(x, (tuple, list)) and not hasattr(x, "shape"):
            return None
        return tuple(cast("Callable[..., list[int]]", _get_shape_list)(x))


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
            tuple[int, ...]: Result.
        """
        x = args[0] if args else kwargs.get("m", kwargs.get("x"))
        if not isinstance(x, (tuple, list)) and not hasattr(x, "shape"):
            return None
        return tuple(cast("Callable[..., list[int]]", _get_shape_list)(x))


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
            tuple[int, ...]: Result.
        """
        x = args[0] if args else kwargs.get("m", kwargs.get("x"))
        if not isinstance(x, (tuple, list)) and not hasattr(x, "shape"):
            return None
        return tuple(cast("Callable[..., list[int]]", _get_shape_list)(x))
