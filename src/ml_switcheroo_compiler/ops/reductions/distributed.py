"""Reductions."""

from __future__ import annotations

from ml_switcheroo_compiler.ops.base import register_op
from ml_switcheroo_compiler.ops.reductions.core import ReductionOp


@register_op("Psum")
class Psum(ReductionOp):
    """Parallel sum reduction operation."""

    op_name = "Psum"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer the output shape for the infer_shape operation.

        Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

        Returns:
        object: Result.
        """
        x = args[0] if len(args) > 0 else kwargs.get("x")
        return getattr(x, "shape", ())


@register_op("Pmean")
class Pmean(ReductionOp):
    """Parallel mean reduction operation."""

    op_name = "Pmean"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
        """
        x = args[0] if len(args) > 0 else kwargs.get("x")
        return getattr(x, "shape", ())
