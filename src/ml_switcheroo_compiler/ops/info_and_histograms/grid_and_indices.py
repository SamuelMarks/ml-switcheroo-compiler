# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Misc operations."""

from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("Indices")
class Indices(OpDef):
    """Return an array representing the indices of a grid."""

    op_name: object = "Indices"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        if not args:
            return ()
        dimensions: object = args[0]
        dim_len: object = len(dimensions) if isinstance(dimensions, (list, tuple)) else 0
        return (dim_len, *dimensions) if dim_len > 0 else ()


@register_op("Ix")
class Ix(OpDef):
    """Construct an open mesh from multiple sequences."""

    op_name: object = "Ix"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        nd: object = len(args)
        if nd == 0:
            return ()
        # ix_ returns a tuple of ndarrays, each having ndim == nd. We return shape of the first output.
        # Actually it returns a tuple of arrays, the Op should maybe return a tuple of shapes,
        # but since we can only return one shape, let's return the shape of the first one.
        shape: object = [1] * nd
        if hasattr(args[0], "shape") and len(args[0].shape) > 0:
            shape[0] = args[0].shape[0]
        return tuple(shape)


@register_op("MaskIndices")
class MaskIndices(OpDef):
    """Return the indices to access (n, n) arrays."""

    op_name: object = "MaskIndices"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return (None,)


@register_op("Mgrid")
class Mgrid(OpDef):
    """nd_grid instance which returns a dense multi-dimensional 'meshgrid'."""

    op_name: object = "Mgrid"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return kwargs.get("shape", ())


@register_op("Ogrid")
class Ogrid(OpDef):
    """nd_grid instance which returns an open multi-dimensional 'meshgrid'."""

    op_name: object = "Ogrid"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return kwargs.get("shape", ())


@register_op("R")
class R(OpDef):
    """Translate slice objects to concatenation along the first axis."""

    op_name: object = "R"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return (None,)


def mgrid(*args: object, **kwargs: object) -> object:
    """nd_grid instance which returns a dense multi-dimensional 'meshgrid'.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Mgrid", *args, **kwargs)


def ogrid(*args: object, **kwargs: object) -> object:
    """nd_grid instance which returns an open multi-dimensional 'meshgrid'.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Ogrid", *args, **kwargs)


def r_(*args: object, **kwargs: object) -> object:
    """Translate slice objects to concatenation along the first axis.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("R", *args, **kwargs)


def indices(*args: object, **kwargs: object) -> object:
    """Return an array representing the indices of a grid.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Indices", *args, **kwargs)


def ix_(*args: object, **kwargs: object) -> object:
    """Construct an open mesh from multiple sequences.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Ix", *args, **kwargs)


def mask_indices(*args: object, **kwargs: object) -> object:
    """Return the indices to access (n, n) arrays.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("MaskIndices", *args, **kwargs)
