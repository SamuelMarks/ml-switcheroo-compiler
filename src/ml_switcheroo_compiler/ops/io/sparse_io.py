"""Sparse activation and elementwise I/O operations."""

from __future__ import annotations

from ml_switcheroo_compiler.core.shape import (
    OpValue,
    ShapeProtocol,
    extract_dtype,
    extract_shape,
)
from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("SparsePlus")
class SparsePlus(OpDef):
    """SparsePlus operation definition."""

    op_name = "SparsePlus"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer broadcasted output shape for SparsePlus.

        Args:
            *args (OpValue): Positional tensor arguments.
            **kwargs (OpValue): Keyword operation attributes.

        Returns:
            OpValue: Inferred output shape tuple.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [extract_shape(a) for a in args if isinstance(a, ShapeProtocol)]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        _typed_ret_1: OpValue = res
        return _typed_ret_1


@register_op("SparseSigmoid")
class SparseSigmoid(OpDef):
    """SparseSigmoid operation definition."""

    op_name = "SparseSigmoid"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer broadcasted output shape for SparseSigmoid.

        Args:
            *args (OpValue): Positional tensor arguments.
            **kwargs (OpValue): Keyword operation attributes.

        Returns:
            OpValue: Inferred output shape tuple.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [extract_shape(a) for a in args if isinstance(a, ShapeProtocol)]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        _typed_ret_2: OpValue = res
        return _typed_ret_2


def sparse_plus(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Compute the sparse plus activation function.

    Args:
        *args (OpValue): Positional input tensors.
        **kwargs (OpValue): Keyword operation arguments.

    Returns:
        OpValue: Resulting tensor after applying sparse plus.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("SparsePlus", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node(
        "SparsePlus",
        list(args),
        kwargs,
        extract_shape(args[0]) if args else (),
        extract_dtype(args[0]) if args else "float32",
    )


def sparse_sigmoid(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Compute the sparse sigmoid activation function.

    Args:
        *args (OpValue): Positional input tensors.
        **kwargs (OpValue): Keyword operation arguments.

    Returns:
        OpValue: Resulting tensor after applying sparse sigmoid.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("SparseSigmoid", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node(
        "SparseSigmoid",
        list(args),
        kwargs,
        extract_shape(args[0]) if args else (),
        extract_dtype(args[0]) if args else "float32",
    )
