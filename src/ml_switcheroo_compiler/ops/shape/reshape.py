# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
# pylint: disable=duplicate-code

"""Define shape manipulation operations for the ML Switcheroo framework."""

from ml_switcheroo_compiler.core.constants import MAGIC_VAL_3
from ml_switcheroo_compiler.core.shape import broadcast_shapes
from ml_switcheroo_compiler.ops.base import OpDef, register_op


def _get_shape_list(x):
    """Help to convert shape to list.

    Args:
        x (object): The x parameter.

    Returns:
        list: Result.
    """
    if isinstance(x, (tuple, list)):
        return list(x)
    if not hasattr(x, "shape") or x.shape is None:
        return []
    s = x.shape
    if isinstance(s, tuple):
        return list(s)
    if isinstance(s, int):
        return [s]
    if isinstance(s, list):
        return list(s)
    return []


def _normalize_axes(axes, length: int):
    """Help to normalize axes.

    Args:
        axes (object): The axes parameter.
        length (int): The length parameter.

    Returns:
        list: Result.
    """
    a = [axes] if isinstance(axes, int) else axes
    return [x + length if x < 0 else x for x in a]


def _resolve_reshape_minus_one(x_shape, newshape):
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

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        x = args[0] if args else kwargs.get("a", kwargs.get("x"))
        newshape = args[1] if len(args) > 1 else kwargs["newshape"]

        if not hasattr(x, "shape") or not isinstance(newshape, (tuple, list)):
            return newshape

        out_shape = list(newshape)
        if -1 in out_shape:
            out_shape = _resolve_reshape_minus_one(x.shape, out_shape)

        return tuple(out_shape)


@register_op("Transpose")
class Transpose(OpDef):
    """Provide an operator definition for transposing the dimensions of a tensor."""

    def infer_shape(self, *args, **kwargs):
        """Infer the output shape of the operation.

        Args:
            *args (object): The first input tensor and the axes.
            **kwargs (object): Additional keyword arguments.

        Returns:
            The computed shape or evaluation result.
        """
        x = args[0] if len(args) > 0 else kwargs["x"]
        axes = args[1] if len(args) > 1 else kwargs.get("axes", None)
        if isinstance(x, tuple) and axes is not None:
            return tuple(x[i] for i in axes)
        return tuple(reversed(x)) if isinstance(x, tuple) else None

    def _format_args(self, x: str, axes) -> str:
        """Evaluate _format_args operation.

        Args:
        x (str): The x parameter.
        axes (object): The axes parameter.

        Returns:
        str: Result.
        """
        return f"{x}" if axes is None else f"{x}, {axes}"


@register_op("BroadcastTo")
class BroadcastTo(OpDef):
    """Provide an operator definition for broadcasting a tensor to a new shape."""

    def infer_shape(self, x, shape, **kwargs):
        """Infer the output shape of the operation.

        Args:
            x (object): The x parameter.
            shape (object): The shape parameter.
            **kwargs (object): Keyword args.

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
        x,
        shape,
        broadcast_dimensions,
        **kwargs,
    ):
        """Infer the output shape of the broadcasting operation.

        Args:
            x (object): The input x tensor.
            shape (object): The target shape.
            broadcast_dimensions (object): The broadcast_dimensions parameter for the operation.
            **kwargs (object): Additional keyword arguments.

        Returns: Tensor: The computed result.
        """
        return tuple(shape)


@register_op("Resize")
class Resize(OpDef):
    """Provide an operator definition for resizing an image tensor to a target shape."""

    op_name = "Resize"

    def infer_shape(
        self,
        image,
        shape,
        method="bilinear",
        **kwargs,
    ):
        """Infer the output shape of the resizing operation.

        Args:
            image (object): The image parameter for the operation.
            shape (object): The target shape.
            method (object): The method parameter for the operation.
            **kwargs (object): Additional keyword arguments.

        Returns: Tensor: The computed result.
        """
        if not hasattr(image, "shape") or not image.shape:
            return ()
        out_shape = list(image.shape)
        # assuming shape is (new_h, new_w) and image is either (..., H, W, C) or something.
        # Often it's (..., H, W, C) in Keras.
        if len(out_shape) >= MAGIC_VAL_3:
            out_shape[-3] = shape[0]
            out_shape[-2] = shape[1]
        return tuple(out_shape)


@register_op("Atleast1d")
class Atleast1d(OpDef):
    """Provide an operator definition for atleast_1d."""

    op_name = "Atleast1d"

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        x = args[0] if len(args) > 0 else kwargs.get("a", kwargs.get("x"))
        if not hasattr(x, "shape") or x.shape is None:
            return (1,)
        shape = _get_shape_list(x)
        if len(shape) < 1:
            shape.insert(0, 1)
        return tuple(shape)


@register_op("Atleast2d")
class Atleast2d(OpDef):
    """Provide an operator definition for atleast_2d."""

    op_name = "Atleast2d"

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        x = args[0] if len(args) > 0 else kwargs.get("a", kwargs.get("x"))
        if not hasattr(x, "shape") or x.shape is None:
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

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        x = args[0] if len(args) > 0 else kwargs.get("a", kwargs.get("x"))
        if not hasattr(x, "shape") or x.shape is None:
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

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        x = args[0] if len(args) > 0 else kwargs.get("a", kwargs.get("x"))
        axis = args[1] if len(args) > 1 else kwargs.get("axis")
        if not hasattr(x, "shape") or x.shape is None:
            return (None,)
        shape = _get_shape_list(x)
        if axis is None:
            return tuple(shape)
        if axis < 0:
            axis = axis + len(shape) + 1
        shape.insert(axis, 1)
        return tuple(shape)


@register_op("Block")
class Block(OpDef):
    """Block operator."""

    op_name = "Block"

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

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

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return ()


@register_op("Collapse")
class Collapse(OpDef):
    """Collapse operation for shape manipulation."""

    op_name = "Collapse"

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return ()


@register_op("Delete")
class Delete(OpDef):
    """Delete operator."""

    op_name = "Delete"

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

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

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        n = args[0] if args else kwargs.get("n")
        ndim = args[1] if len(args) > 1 else kwargs.get("ndim", 2)
        return tuple([(n,)] * ndim)


@register_op("DiagIndicesFrom")
class DiagIndicesFrom(OpDef):
    """DiagIndicesFrom operator."""

    op_name = "DiagIndicesFrom"

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        arr = args[0] if args else kwargs.get("arr")
        if arr is None or not hasattr(arr, "shape"):
            return None
        return tuple([(arr.shape[0],)] * len(arr.shape))


@register_op("Diagflat")
class Diagflat(OpDef):
    """Diagflat operator."""

    op_name = "Diagflat"

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        v = args[0] if args else kwargs.get("v")
        if not hasattr(v, "shape"):
            return None
        import math

        size = math.prod(v.shape)
        k = args[1] if len(args) > 1 else kwargs.get("k", 0)
        s = size + abs(k)
        return (s, s)


@register_op("FillDiagonal")
class FillDiagonal(OpDef):
    """FillDiagonal operator."""

    op_name = "FillDiagonal"

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        a = args[0] if args else kwargs.get("a")
        if not hasattr(a, "shape"):
            return None
        return tuple(a.shape)


@register_op("Insert")
class Insert(OpDef):
    """Insert operator."""

    op_name = "Insert"

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

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

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        x = args[0] if args else kwargs.get("a", kwargs.get("x"))
        if not hasattr(x, "shape") or x.shape is None:
            return None
        shape = _get_shape_list(x)
        source = args[1] if len(args) > 1 else kwargs.get("source")
        destination = args[2] if len(args) > 2 else kwargs.get("destination")
        return self._calc_shape(shape, source, destination)

    def _calc_shape(self, shape: list[int], source, destination):
        """Calculate the output shape for a moveaxis operation.

        Args:
            shape (list[int]): The input shape.
            source (object): The source axes.
            destination (object): The destination axes.

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

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        x = args[0] if args else kwargs.get("a", kwargs.get("x"))
        if not isinstance(x, (tuple, list)) and not hasattr(x, "shape"):
            return None
        shape = _get_shape_list(x)
        dims = args[1] if len(args) > 1 else kwargs.get("dims")
        if dims is None:
            return tuple(shape[::-1])
        return tuple(shape[i] for i in dims)


@register_op("Roll")
class Roll(OpDef):
    """Provide an operator definition for rolling array elements along a given axis."""

    op_name = "Roll"

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        x = args[0] if args else kwargs.get("a", kwargs.get("x"))
        if not isinstance(x, (tuple, list)) and not hasattr(x, "shape"):
            return None
        return tuple(_get_shape_list(x))


@register_op("Squeeze")
class Squeeze(OpDef):
    """Provide an operator definition for squeezing dimensions of a tensor."""

    op_name = "Squeeze"

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        x = args[0] if args else kwargs.get("a", kwargs.get("x"))
        if not isinstance(x, (tuple, list)) and not hasattr(x, "shape"):
            return None
        shape = _get_shape_list(x)
        axis = args[1] if len(args) > 1 else kwargs.get("axis")
        return self._calc_shape(shape, axis)

    def _calc_shape(self, shape: list[int], axis):
        """Calculate the output shape for a squeeze operation.

        Args:
            shape (list[int]): The input shape.
            axis (object): The axis or axes to squeeze.

        Returns:
            tuple: The calculated shape.
        """
        if axis is None:
            return tuple(s for s in shape if s != 1)
        axes = [axis] if isinstance(axis, int) else axis
        n = {a + len(shape) if a < 0 else a for a in axes}
        return tuple(shape[i] for i in range(len(shape)) if i not in n)


@register_op("Swapaxes")
class Swapaxes(OpDef):
    """Provide an operator definition for swapping two axes of an array."""

    op_name = "Swapaxes"

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        x = args[0] if args else kwargs.get("a", kwargs.get("x"))
        if not isinstance(x, (tuple, list)) and not hasattr(x, "shape"):
            return None
        shape = _get_shape_list(x)
        axis1 = args[1] if len(args) > 1 else kwargs.get("axis1")
        axis2 = args[2] if len(args) > 2 else kwargs.get("axis2")
        a1 = axis1 + len(shape) if axis1 < 0 else axis1
        a2 = axis2 + len(shape) if axis2 < 0 else axis2
        shape[a1], shape[a2] = shape[a2], shape[a1]
        return tuple(shape)


@register_op("Flip")
class Flip(OpDef):
    """Provide an operator definition for flipping an array."""

    op_name = "Flip"

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        x = args[0] if args else kwargs.get("m", kwargs.get("x"))
        if not isinstance(x, (tuple, list)) and not hasattr(x, "shape"):
            return None
        return tuple(_get_shape_list(x))


@register_op("Fliplr")
class Fliplr(OpDef):
    """Provide an operator definition for flipping an array left/right."""

    op_name = "Fliplr"

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        x = args[0] if args else kwargs.get("m", kwargs.get("x"))
        if not isinstance(x, (tuple, list)) and not hasattr(x, "shape"):
            return None
        return tuple(_get_shape_list(x))


@register_op("Flipud")
class Flipud(OpDef):
    """Provide an operator definition for flipping an array up/down."""

    op_name = "Flipud"

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        x = args[0] if args else kwargs.get("m", kwargs.get("x"))
        if not isinstance(x, (tuple, list)) and not hasattr(x, "shape"):
            return None
        return tuple(_get_shape_list(x))
