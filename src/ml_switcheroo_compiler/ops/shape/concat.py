# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
# pylint: disable=duplicate-code

"""Define shape manipulation operations for the ML Switcheroo framework."""

from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("Concatenate")
class Concatenate(OpDef):
    """Concatenate operator definition.

    This operator concatenates a sequence of arrays along an existing axis.
    """

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        """Infer the output shape for the Concatenate operation.

        Args:
            *args: Positional arguments containing the input shapes to concatenate.
            **kwargs: Keyword arguments containing configuration like the axis.

        Returns:
            tuple[int, ...]: The inferred shape of the concatenated output tensor.
        """
        if len(args) > 0 and isinstance(args[0], (list, tuple)):
            shapes = args[0]
            axis = kwargs.get("axis", 0)
            if not shapes:
                return ()
            res = list(shapes[0])
            res[axis] = sum(s[axis] if len(s) > axis else 1 for s in shapes)
            return tuple(res)
        return ()


@register_op("Stack")
class Stack(OpDef):
    """Stack operator definition.

    This operator joins a sequence of arrays along a new axis.
    """

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple: Result.
        """
        """Infer the output shape for the Stack operation.

        Args:
            *args: Positional arguments containing the input shapes to stack.
            **kwargs: Keyword arguments containing configuration like the axis.

        Returns:
            tuple[int, ...]: The inferred shape of the stacked output tensor.
        """
        return ()


@register_op("Split")
class Split(OpDef):
    """Split operator definition.

    This operator splits an array into multiple sub-arrays.
    """

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple: Result.
        """
        """Infer the output shape for the Split operation.

        Args:
            *args: Positional arguments containing the input shape to split.
            **kwargs: Keyword arguments containing configuration like indices or sections.

        Returns:
            tuple[int, ...]: The inferred shape of the split output tensors.
        """
        if len(args) > 0 and hasattr(args[0], "shape"):
            return tuple(args[0].shape)
        return ()


@register_op("Hsplit")
class Hsplit(OpDef):
    """Hsplit operator definition.

    This operator splits an array into multiple sub-arrays horizontally (column-wise).
    """

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple: Result.
        """
        """Infer the output shape for the Hsplit operation.

        Args:
            *args: Positional arguments containing the input shape to split horizontally.
            **kwargs: Keyword arguments containing configuration for the split.

        Returns:
            tuple[int, ...]: The inferred shape of the split output tensors.
        """
        if len(args) > 0 and hasattr(args[0], "shape"):
            return tuple(args[0].shape)
        return ()


@register_op("Vsplit")
class Vsplit(OpDef):
    """Vsplit operator definition.

    This operator splits an array into multiple sub-arrays vertically (row-wise).
    """

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple: Result.
        """
        """Infer the output shape for the Vsplit operation.

        Args:
            *args: Positional arguments containing the input shape to split vertically.
            **kwargs: Keyword arguments containing configuration for the split.

        Returns:
            tuple[int, ...]: The inferred shape of the split output tensors.
        """
        if len(args) > 0 and hasattr(args[0], "shape"):
            return tuple(args[0].shape)
        return ()


@register_op("Dsplit")
class Dsplit(OpDef):
    """Dsplit operator definition.

    This operator splits an array into multiple sub-arrays along the 3rd axis (depth).
    """

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple: Result.
        """
        """Infer the output shape for the Dsplit operation.

        Args:
            *args: Positional arguments containing the input shape to split along depth.
            **kwargs: Keyword arguments containing configuration for the split.

        Returns:
            tuple[int, ...]: The inferred shape of the split output tensors.
        """
        if len(args) > 0 and hasattr(args[0], "shape"):
            return tuple(args[0].shape)
        return ()


@register_op("Hstack")
class Hstack(OpDef):
    """Hstack operator definition.

    This operator stacks arrays in sequence horizontally (column wise).
    """

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple: Result.
        """
        if len(args) > 0 and isinstance(args[0], (list, tuple)):
            shapes = [s.shape if hasattr(s, "shape") else s for s in args[0]]
            if not shapes:
                return ()
            # Hstack: if 1D, sum sizes. if >1D, sum along axis 1.
            if len(shapes[0]) == 1:
                return (sum(s[0] for s in shapes),)
            res = list(shapes[0])
            res[1] = sum(s[1] for s in shapes)
            return tuple(res)
        return ()


@register_op("Vstack")
class Vstack(OpDef):
    """Vstack operator definition.

    This operator stacks arrays in sequence vertically (row wise).
    """

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple: Result.
        """
        if len(args) > 0 and isinstance(args[0], (list, tuple)):
            shapes = [s.shape if hasattr(s, "shape") else s for s in args[0]]
            if not shapes:
                return ()
            # Vstack: if 1D, stack as rows => (N, M). if >1D, sum along axis 0.
            if len(shapes[0]) == 1:
                return (len(shapes), shapes[0][0])
            res = list(shapes[0])
            res[0] = sum(s[0] for s in shapes)
            return tuple(res)
        return ()


@register_op("Dstack")
class Dstack(OpDef):
    """Dstack operator definition.

    This operator stacks arrays in sequence depth wise (along third axis).
    """

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple: Result.
        """
        if len(args) > 0 and isinstance(args[0], (list, tuple)):
            shapes = [s.shape if hasattr(s, "shape") else s for s in args[0]]
            if not shapes:
                return ()
            # Dstack: if 1D => (1, M, N). if 2D => (N, M, P). if 3D, sum along axis 2.
            if len(shapes[0]) == 1:
                return (1, shapes[0][0], len(shapes))
            if len(shapes[0]) == 2:
                return (shapes[0][0], shapes[0][1], len(shapes))
            res = list(shapes[0])
            res[2] = sum(s[2] for s in shapes)
            return tuple(res)
        return ()


@register_op("ColumnStack")
class ColumnStack(OpDef):
    """ColumnStack operator definition.

    This operator stacks 1-D arrays as columns into a 2-D array.
    """

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple: Result.
        """
        if len(args) > 0 and isinstance(args[0], (list, tuple)):
            shapes = [s.shape if hasattr(s, "shape") else s for s in args[0]]
            if not shapes:
                return ()
            # ColumnStack: 1D arrays are treated as 2D columns (N, 1) and stacked horizontally.
            if len(shapes[0]) == 1:
                return (shapes[0][0], len(shapes))
            res = list(shapes[0])
            res[1] = sum(s[1] for s in shapes)
            return tuple(res)
        return ()


@register_op("RowStack")
class RowStack(OpDef):
    """RowStack operator definition.

    This operator stacks arrays in sequence vertically (row wise).
    """

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple: Result.
        """
        """Infer the output shape for the RowStack operation.

        Args:
            *args: Positional arguments containing the input shapes to stack as rows.
            **kwargs: Keyword arguments containing configuration for the stack.

        Returns:
            tuple[int, ...]: The inferred shape of the row-stacked output tensor.
        """
        return ()


@register_op("Argwhere")
class Argwhere(OpDef):
    """Find the indices of array elements that are non-zero, grouped by element."""

    op_name = "Argwhere"
    np_op_name = "argwhere"

    def infer_shape(self, a, **kwargs):
        """Infer the output shape for the Argwhere operation.

        Args:
            a: The input array or shape whose non-zero elements are to be found.
            **kwargs: Keyword arguments containing configuration for the operation.

        Returns: Tensor: The inferred shape of the output indices tensor.
        """
        return (None, len(a) if isinstance(a, tuple) else None)


@register_op("Append")
class Append(OpDef):
    """Append operator definition.

    This operator appends values to the end of an array.
    """

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple: Result.
        """
        if not args or not hasattr(args[0], "shape"):
            return ()
        axis = kwargs.get("axis")
        if axis is not None:
            res = list(args[0].shape)
            res[axis] += args[1].shape[axis] if len(args) > 1 and hasattr(args[1], "shape") else 1
            return tuple(res)
        import math

        s1 = math.prod(args[0].shape)
        s2 = math.prod(args[1].shape) if len(args) > 1 and hasattr(args[1], "shape") else 1
        return (s1 + s2,)
