"""Misc operations."""

from __future__ import annotations

import typing
from collections.abc import Sequence

from ml_switcheroo_compiler.backends.registry import EagerTensorProtocol
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("Indices")
class Indices(OpDef):
    """Return an array representing the indices of a grid."""

    op_name = "Indices"

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            *args (EagerTensorProtocol): Positional args.
            **kwargs (EagerTensorProtocol): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        if not args:
            return ()
        dimensions = args[0]
        dim_len = len(dimensions) if isinstance(dimensions, (list, tuple)) else 0
        return (dim_len, *typing.cast("Sequence[int]", dimensions)) if dim_len > 0 else ()


@register_op("Ix")
class Ix(OpDef):
    """Construct an open mesh from multiple sequences."""

    op_name = "Ix"

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            *args (EagerTensorProtocol): Positional args.
            **kwargs (EagerTensorProtocol): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        nd = len(args)
        if nd == 0:
            return ()
        # ix_ returns a tuple of ndarrays, each having ndim == nd. We return shape of the first output.
        # Actually it returns a tuple of arrays, the Op should maybe return a tuple of shapes,
        # but since we can only return one shape, let's return the shape of the first one.
        shape = [1] * nd
        if hasattr(args[0], "shape") and len(args[0].shape) > 0:
            shape[0] = args[0].shape[0]
        return tuple(shape)


@register_op("MaskIndices")
class MaskIndices(OpDef):
    """Return the indices to access (n, n) arrays."""

    op_name = "MaskIndices"

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            *args (EagerTensorProtocol): Positional args.
            **kwargs (EagerTensorProtocol): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return (None,)


@register_op("Mgrid")
class Mgrid(OpDef):
    """nd_grid instance which returns a dense multi-dimensional 'meshgrid'."""

    op_name = "Mgrid"

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            *args (EagerTensorProtocol): Positional args.
            **kwargs (EagerTensorProtocol): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return kwargs.get("shape", ())


@register_op("Ogrid")
class Ogrid(OpDef):
    """nd_grid instance which returns an open multi-dimensional 'meshgrid'."""

    op_name = "Ogrid"

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            *args (EagerTensorProtocol): Positional args.
            **kwargs (EagerTensorProtocol): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return kwargs.get("shape", ())


@register_op("R")
class R(OpDef):
    """Translate slice Anys to concatenation along the first axis."""

    op_name = "R"

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            *args (EagerTensorProtocol): Positional args.
            **kwargs (EagerTensorProtocol): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return (None,)


def mgrid(*args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """nd_grid instance which returns a dense multi-dimensional 'meshgrid'.

    Args:
        *args (TensorOrScalar): Positional args.
        **kwargs (TensorOrScalar): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Mgrid", *args, **kwargs)


def ogrid(*args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """nd_grid instance which returns an open multi-dimensional 'meshgrid'.

    Args:
        *args (TensorOrScalar): Positional args.
        **kwargs (TensorOrScalar): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Ogrid", *args, **kwargs)


def r_(*args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """Translate slice Anys to concatenation along the first axis.

    Args:
        *args (TensorOrScalar): Positional args.
        **kwargs (TensorOrScalar): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("R", *args, **kwargs)


def indices(*args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """Return an array representing the indices of a grid.

    Args:
        *args (TensorOrScalar): Positional args.
        **kwargs (TensorOrScalar): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Indices", *args, **kwargs)


def ix_(*args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """Construct an open mesh from multiple sequences.

    Args:
        *args (TensorOrScalar): Positional args.
        **kwargs (TensorOrScalar): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Ix", *args, **kwargs)


def mask_indices(*args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """Return the indices to access (n, n) arrays.

    Args:
        *args (TensorOrScalar): Positional args.
        **kwargs (TensorOrScalar): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("MaskIndices", *args, **kwargs)
