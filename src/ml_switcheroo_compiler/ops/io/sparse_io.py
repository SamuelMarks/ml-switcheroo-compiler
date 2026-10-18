"""Sparse activation and elementwise I/O operations."""

from __future__ import annotations

from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("SparsePlus")
class SparsePlus(OpDef):
    """SparsePlus operation definition."""

    op_name = "SparsePlus"

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer broadcasted output shape for SparsePlus.

        Args:
            *args (TensorOrScalar): Positional tensor arguments.
            **kwargs (TensorOrScalar): Keyword operation attributes.

        Returns:
            TensorOrScalar: Inferred output shape tuple.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        return res


@register_op("SparseSigmoid")
class SparseSigmoid(OpDef):
    """SparseSigmoid operation definition."""

    op_name = "SparseSigmoid"

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer broadcasted output shape for SparseSigmoid.

        Args:
            *args (TensorOrScalar): Positional tensor arguments.
            **kwargs (TensorOrScalar): Keyword operation attributes.

        Returns:
            TensorOrScalar: Inferred output shape tuple.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        return res


def sparse_plus(*args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """Compute the sparse plus activation function.

    Args:
        *args (TensorOrScalar): Positional input tensors.
        **kwargs (TensorOrScalar): Keyword operation arguments.

    Returns:
        TensorOrScalar: Resulting tensor after applying sparse plus.
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
        getattr(args[0], "shape", ()) if args else (),
        getattr(args[0], "dtype", "float32") if args else "float32",
    )


def sparse_sigmoid(*args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """Compute the sparse sigmoid activation function.

    Args:
        *args (TensorOrScalar): Positional input tensors.
        **kwargs (TensorOrScalar): Keyword operation arguments.

    Returns:
        TensorOrScalar: Resulting tensor after applying sparse sigmoid.
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
        getattr(args[0], "shape", ()) if args else (),
        getattr(args[0], "dtype", "float32") if args else "float32",
    )
