"""Dot product operations."""

from __future__ import annotations

from ml_switcheroo_compiler.ops.base import OpDef, register_op


def _has_valid_shape(obj: object) -> bool:
    """Evaluate _has_valid_shape operation.

    Args:
        obj (object): The obj parameter.

    Returns:
        bool: Result.
    """
    return hasattr(obj, "shape") and bool(obj.shape)


@register_op("Dot")
class Dot(OpDef):
    """Dot product operator.

    Computes the dot product of two arrays
    """

    def infer_shape(self, a: object, b: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            a (object): The a parameter.
            b (object): The b parameter.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
        """
        return None


@register_op("DotGeneral")
class DotGeneral(OpDef):
    """General dot product operator.

    Computes a generalized dot product matching JAX's lax.dot_general.
    """

    op_name = "DotGeneral"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        return res

    def _compute_out_shape(self, lhs_shape: tuple, rhs_shape: tuple, dimension_numbers: tuple) -> tuple:
        """Evaluate _compute_out_shape operation.

        Args:
            lhs_shape (tuple): The lhs_shape parameter.
            rhs_shape (tuple): The rhs_shape parameter.
            dimension_numbers (tuple): The dimension_numbers parameter.

        Returns:
            tuple: Result.
        """
        contracting, batch = dimension_numbers
        lhs_contracting, rhs_contracting = contracting
        lhs_batch, rhs_batch = batch

        out_shape = [lhs_shape[b] for b in lhs_batch]
        out_shape.extend([lhs_shape[i] for i in range(len(lhs_shape)) if i not in lhs_contracting + lhs_batch])
        out_shape.extend([rhs_shape[i] for i in range(len(rhs_shape)) if i not in rhs_contracting + rhs_batch])

        return tuple(out_shape)


@register_op("Tensordot")
class Tensordot(OpDef):
    """Tensordot operator.

    Computes tensor dot product along specified axes.
    """

    def infer_shape(self, a: object, b: object, **kwargs: object) -> object:
        """Infer the output shape of the operation.

        Args:
            a (object): The first input tensor.
            b (object): The second input tensor.
            **kwargs (object): Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return ()


@register_op("Inner")
class Inner(OpDef):
    """Inner product operator.

    Computes the inner product of two vectors or matrices.
    """

    def infer_shape(self, a: object, b: object, **kwargs: object) -> object:
        """Infer the output shape of the operation.

        Args:
            a (object): The first input tensor.
            b (object): The second input tensor.
            **kwargs (object): Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return ()


@register_op("Outer")
class Outer(OpDef):
    """Outer product operator.

    Computes the outer product of two vectors.
    """

    def infer_shape(self, a: object, b: object, **kwargs: object) -> object:
        """Infer the output shape of the operation.

        Args:
            a (object): The first input tensor.
            b (object): The second input tensor.
            **kwargs (object): Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return ()


def _compute_pdot_shape(lhs_shape: tuple, rhs_shape: tuple) -> tuple:
    """Evaluate _compute_pdot_shape operation.

    Args:
        lhs_shape (tuple): The lhs_shape parameter.
        rhs_shape (tuple): The rhs_shape parameter.

    Returns:
        tuple: Result.
    """
    if len(lhs_shape) == 1 and len(rhs_shape) == 1:
        return ()
    elif len(lhs_shape) == 2 and len(rhs_shape) == 2:
        return (lhs_shape[0], rhs_shape[1])
    elif len(lhs_shape) == 0 or len(rhs_shape) == 0:
        # scalar multiplication
        return lhs_shape if len(lhs_shape) > 0 else rhs_shape
    elif len(rhs_shape) == 1:
        return lhs_shape[:-1]
    else:
        return lhs_shape[:-1] + rhs_shape[:-2] + rhs_shape[-1:]


@register_op("Pdot")
class Pdot(OpDef):
    """Parallel dot product operator."""

    op_name = "Pdot"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
        """
        lhs = args[0] if len(args) > 0 else None
        rhs = args[1] if len(args) > 1 else None
        lhs_shape = getattr(lhs, "shape", ())
        rhs_shape = getattr(rhs, "shape", ())
        return _compute_pdot_shape(lhs_shape, rhs_shape)


def pdot(*args: object, **kwargs: object) -> object:
    """Evaluate pdot operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Pdot", *args, **kwargs)
