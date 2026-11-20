"""Misc operations."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.registry import coerce_shape
from ml_switcheroo_compiler.core.shape import OpValue, extract_shape
from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("Indices")
class Indices(OpDef):
    """Return an array representing the indices of a grid."""

    op_name = "Indices"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        if not args:
            return ()
        dimensions = args[0]
        dim_len = len(dimensions) if isinstance(dimensions, (list, tuple)) else 0
        return (dim_len, *coerce_shape(dimensions)) if dim_len > 0 else ()


@register_op("Ix")
class Ix(OpDef):
    """Construct an open mesh from multiple sequences."""

    op_name = "Ix"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        nd = len(args)
        if nd == 0:
            return ()
        # ix_ returns a tuple of ndarrays, each having ndim == nd. We return shape of the first output.
        # Actually it returns a tuple of arrays, the Op should maybe return a tuple of shapes,
        # but since we can only return one shape, let's return the shape of the first one.
        shape = [1] * nd
        first_shape = extract_shape(args[0])
        if len(first_shape) > 0:
            shape[0] = first_shape[0]
        return tuple(shape)


@register_op("MaskIndices")
class MaskIndices(OpDef):
    """Return the indices to access (n, n) arrays."""

    op_name = "MaskIndices"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        return (None,)


@register_op("Mgrid")
class Mgrid(OpDef):
    """nd_grid instance which returns a dense multi-dimensional 'meshgrid'."""

    op_name = "Mgrid"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        return kwargs.get("shape", ())


@register_op("Ogrid")
class Ogrid(OpDef):
    """nd_grid instance which returns an open multi-dimensional 'meshgrid'."""

    op_name = "Ogrid"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        return kwargs.get("shape", ())


@register_op("R")
class R(OpDef):
    """Translate slice objects to concatenation along the first axis."""

    op_name = "R"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        return (None,)


def mgrid(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """nd_grid instance which returns a dense multi-dimensional 'meshgrid'.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Mgrid", *args, **kwargs)


def ogrid(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """nd_grid instance which returns an open multi-dimensional 'meshgrid'.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Ogrid", *args, **kwargs)


def r_(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Translate slice objects to concatenation along the first axis.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("R", *args, **kwargs)


def indices(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Return an array representing the indices of a grid.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Indices", *args, **kwargs)


def ix_(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Construct an open mesh from multiple sequences.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Ix", *args, **kwargs)


def mask_indices(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Return the indices to access (n, n) arrays.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("MaskIndices", *args, **kwargs)
