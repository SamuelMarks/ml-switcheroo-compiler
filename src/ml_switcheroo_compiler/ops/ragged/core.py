"""Ragged ops core."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.registry import EagerTensorProtocol
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("RaggedDot")
class RaggedDot(OpDef):
    """RaggedDot op."""

    op_name = "RaggedDot"

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer the output shape for the RaggedDot operation.

        Args:
            *args (TensorOrScalar): Input tensors (rt_a, b).
            **kwargs (TensorOrScalar): Optional keyword arguments.

        Returns:
            tuple: Result shape (B, var_dim, N).
        """
        from ml_switcheroo_compiler.ir.shape_system import SymVar

        if len(args) >= 2:
            a, b = args[0], args[1]
        else:
            a = typing.cast(EagerTensorProtocol, kwargs.get("rt_a", kwargs.get("a", args[0] if args else None)))
            b = typing.cast(EagerTensorProtocol, kwargs.get("b", kwargs.get("weights", args[1] if len(args) > 1 else None)))

        if a is None:
            return ()
        shape_a = tuple(getattr(a, "shape", getattr(a, "shape_metadata", a if isinstance(a, (list, tuple)) else ())))
        shape_b = tuple(getattr(b, "shape", getattr(b, "shape_metadata", b if isinstance(b, (list, tuple)) else ()))) if b is not None else ()

        batch = shape_a[0] if len(shape_a) > 0 else 1
        ragged_dim = shape_a[1] if len(shape_a) > 1 else SymVar("ragged_dim")
        out_features = shape_b[-1] if len(shape_b) > 0 else (shape_a[-1] if len(shape_a) > 2 else 1)
        return (batch, ragged_dim, out_features)
