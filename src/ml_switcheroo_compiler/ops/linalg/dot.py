"""Dot product operations."""

from __future__ import annotations

import typing
from collections.abc import Sequence

from ml_switcheroo_compiler.backends.registry import EagerTensorProtocol
from ml_switcheroo_compiler.core.shape import TensorOrScalar, TypedVariadicCallable
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.base import OpDef, register_op


def _has_valid_shape(obj: Tensor | Sequence[int] | None) -> bool:
    """Compute the has valid shape tensor operation.

    Args:
        obj: Target tensor or shape sequence.

    Returns:
        bool: Result.
    """
    return hasattr(obj, "shape") and bool(getattr(obj, "shape"))


def _extract_dot_operand_shape(obj: Tensor | Sequence[int] | None) -> tuple[int, ...]:
    """Extract shape tuple from a tensor or shape sequence.

    Args:
        obj: Target tensor or shape sequence.

    Returns:
        tuple[int, ...]: Extracted shape tuple.
    """
    if hasattr(obj, "shape"):
        return tuple(int(d) for d in getattr(obj, "shape"))
    if hasattr(obj, "shape_metadata"):
        sm = getattr(obj, "shape_metadata")
        if sm:
            return tuple(int(d) for d in sm)
    if isinstance(obj, (list, tuple)):
        return tuple(int(d) for d in obj)
    return ()


def _compute_dot_contraction_shape(shape_a: tuple[int, ...], shape_b: tuple[int, ...]) -> tuple[int, ...]:
    """Compute contracted output shape for two operand shape tuples.

    Args:
        shape_a: Shape tuple of first operand.
        shape_b: Shape tuple of second operand.

    Returns:
        tuple[int, ...]: Contracted output shape tuple.
    """
    ndim_a = len(shape_a)
    ndim_b = len(shape_b)
    if ndim_a == 0 or ndim_b == 0:
        return shape_b if ndim_a == 0 else shape_a
    if ndim_b == 1:
        return () if ndim_a == 1 else shape_a[:-1]
    if ndim_a == 1:
        return shape_b[1:]
    if ndim_a == 2 and ndim_b == 2:
        return (shape_a[0], shape_b[1])
    return shape_a[:-1] + shape_b[:-2] + (shape_b[-1],)


@register_op("Dot")
class Dot(OpDef):
    """Dot product operator.

    Computes the dot product of two arrays
    """

    def infer_shape(self, a: TensorOrScalar = None, b: TensorOrScalar = None, **kwargs: TensorOrScalar) -> tuple[int, ...]:
        """Infer output shape for dot product following standard tensor contraction rules.

        Args:
            a (TensorOrScalar): First input tensor or shape.
            b (TensorOrScalar): Second input tensor or shape.
            **kwargs (TensorOrScalar): Optional keyword arguments.

        Returns:
            tuple[int, ...]: Inferred output shape.
        """
        inp_a = a if a is not None else (kwargs.get("a", kwargs.get("lhs")))
        inp_b = b if b is not None else (kwargs.get("b", kwargs.get("rhs")))
        if inp_a is None or inp_b is None:
            return ()

        shape_a = typing.cast("TypedVariadicCallable[tuple[int, ...]]", _extract_dot_operand_shape)(inp_a)
        shape_b = typing.cast("TypedVariadicCallable[tuple[int, ...]]", _extract_dot_operand_shape)(inp_b)
        return _compute_dot_contraction_shape(shape_a, shape_b)


@register_op("DotGeneral")
class DotGeneral(OpDef):
    """General dot product operator.

    Computes a generalized dot product matching JAX's lax.dot_general.
    """

    op_name = "DotGeneral"

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            *args (TensorOrScalar): Positional args.
            **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        return res

    def _compute_out_shape(self, lhs_shape: TensorOrScalar, rhs_shape: TensorOrScalar, dimension_numbers: TensorOrScalar) -> TensorOrScalar:
        """Compute the compute out shape tensor operation.

        Args:
            lhs_shape (tuple): The lhs_shape parameter.
            rhs_shape (tuple): The rhs_shape parameter.
            dimension_numbers (tuple): The dimension_numbers parameter.

        Returns:
            tuple: Result.
        """
        contracting, batch = getattr(dimension_numbers, "__iter__")()
        lhs_contracting, rhs_contracting = getattr(contracting, "__iter__")()
        lhs_batch, rhs_batch = getattr(batch, "__iter__")()

        out_shape = [getattr(lhs_shape, "__getitem__")(b) for b in getattr(lhs_batch, "__iter__")()]
        out_shape.extend([getattr(lhs_shape, "__getitem__")(i) for i in range(getattr(lhs_shape, "__len__")()) if i not in getattr(lhs_contracting + lhs_batch, "__iter__")()])
        out_shape.extend([getattr(rhs_shape, "__getitem__")(i) for i in range(getattr(rhs_shape, "__len__")()) if i not in getattr(rhs_contracting + rhs_batch, "__iter__")()])

        return tuple(out_shape)


@register_op("Tensordot")
class Tensordot(OpDef):
    """Tensordot operator.

    Computes tensor dot product along specified axes.
    """

    def infer_shape(self, a: EagerTensorProtocol, b: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer the output shape of the operation.

        Args:
            a (EagerTensorProtocol): The first input tensor.
            b (EagerTensorProtocol): The second input tensor.
            **kwargs (TensorOrScalar): Additional keyword arguments.

        Returns: Tensor: The computed result.
        """
        return ()


@register_op("Inner")
class Inner(OpDef):
    """Inner product operator.

    Computes the inner product of two vectors or matrices.
    """

    def infer_shape(self, a: EagerTensorProtocol, b: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer the output shape of the operation.

        Args:
            a (EagerTensorProtocol): The first input tensor.
            b (EagerTensorProtocol): The second input tensor.
            **kwargs (TensorOrScalar): Additional keyword arguments.

        Returns: Tensor: The computed result.
        """
        return ()


@register_op("Outer")
class Outer(OpDef):
    """Outer product operator.

    Computes the outer product of two vectors.
    """

    def infer_shape(self, a: EagerTensorProtocol, b: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer the output shape of the operation.

        Args:
            a (EagerTensorProtocol): The first input tensor.
            b (EagerTensorProtocol): The second input tensor.
            **kwargs (TensorOrScalar): Additional keyword arguments.

        Returns: Tensor: The computed result.
        """
        return ()


def _compute_pdot_shape(lhs_shape: tuple[int, ...], rhs_shape: tuple[int, ...]) -> tuple[int, ...]:
    """Compute the compute pdot shape tensor operation.

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

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            *args (TensorOrScalar): Positional args.
            **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        del kwargs
        lhs = args[0]
        rhs = args[1]
        lhs_shape = getattr(lhs, "shape", lhs if isinstance(lhs, tuple) else ())
        rhs_shape = getattr(rhs, "shape", rhs if isinstance(rhs, tuple) else ())
        return _compute_pdot_shape(lhs_shape, rhs_shape)


def pdot(*args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """Compute the pdot tensor operation.

    Args:
        *args (TensorOrScalar): Positional args.
        **kwargs (TensorOrScalar): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Pdot", *args, **kwargs)
