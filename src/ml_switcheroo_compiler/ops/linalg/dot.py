"""Dot product operations."""

from __future__ import annotations

from collections.abc import Sequence

from ml_switcheroo_compiler.backends.registry import EagerTensorProtocol, ensure_eager_list, ensure_shape
from ml_switcheroo_compiler.core.shape import (
    ShapeProtocol,
    TensorOrScalar,
    extract_shape,
)
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.linalg.utils import _extract_linalg_shape


def _has_valid_shape(obj: Tensor | Sequence[int] | None) -> bool:
    """Compute the has valid shape tensor operation.

    Args:
        obj: Target tensor or shape sequence.

    Returns:
        bool: Computed `bool` output.
    """
    raw_obj: TensorOrScalar = obj
    return isinstance(raw_obj, ShapeProtocol) and bool(raw_obj.shape)


def _extract_dot_operand_shape(obj: TensorOrScalar) -> tuple[int, ...]:
    """Extract shape tuple from a tensor or shape sequence.

    Args:
        obj: Target tensor or shape sequence.

    Returns:
        tuple[int, ...]: Extracted shape tuple.
    """
    return _extract_linalg_shape(obj)


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

        shape_a = _extract_dot_operand_shape(inp_a)
        shape_b = _extract_dot_operand_shape(inp_b)
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
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [extract_shape(a) for a in args if isinstance(a, ShapeProtocol)]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        _typed_ret_1: TensorOrScalar = res
        return _typed_ret_1

    def _compute_out_shape(self, lhs_shape: TensorOrScalar, rhs_shape: TensorOrScalar, dimension_numbers: TensorOrScalar) -> TensorOrScalar:
        """Compute the compute out shape tensor operation.

        Args:
            lhs_shape (tuple): Input `lhs_shape` argument (lhs shape).
            rhs_shape (tuple): Input `rhs_shape` argument (rhs shape).
            dimension_numbers (tuple): Input `dimension_numbers` argument (dimension numbers).

        Returns:
            tuple: Computed `tuple` output.
        """
        dim_nums = ensure_eager_list(dimension_numbers)
        contracting, batch = ensure_eager_list(dim_nums[0]), ensure_eager_list(dim_nums[1])
        lhs_contracting, rhs_contracting = ensure_shape(contracting[0]), ensure_shape(contracting[1])
        lhs_batch, rhs_batch = ensure_shape(batch[0]), ensure_shape(batch[1])
        l_shape = ensure_shape(lhs_shape)
        r_shape = ensure_shape(rhs_shape)

        out_shape = [l_shape[b] for b in lhs_batch]
        out_shape.extend([l_shape[i] for i in range(len(l_shape)) if i not in (lhs_contracting + lhs_batch)])
        out_shape.extend([r_shape[i] for i in range(len(r_shape)) if i not in (rhs_contracting + rhs_batch)])

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
        lhs_shape (tuple): Input `lhs_shape` argument (lhs shape).
        rhs_shape (tuple): Input `rhs_shape` argument (rhs shape).

    Returns:
        tuple: Computed `tuple` output.
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
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        del kwargs
        lhs = args[0]
        rhs = args[1]
        lhs_shape = _extract_linalg_shape(lhs)
        rhs_shape = _extract_linalg_shape(rhs)
        return _compute_pdot_shape(lhs_shape, rhs_shape)


def pdot(*args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """Compute the pdot tensor operation.

    Args:
        *args (TensorOrScalar): Positional args.
        **kwargs (TensorOrScalar): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Pdot", *args, **kwargs)
