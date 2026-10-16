"""Reductions."""

from __future__ import annotations

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.base import register_op
from ml_switcheroo_compiler.ops.reductions.core import ReductionOp


@register_op("Psum")
class Psum(ReductionOp):
    """Parallel sum reduction operation."""

    op_name = "Psum"

    def infer_shape(
        self,
        *args: Tensor,
        **kwargs: Tensor | int | float | str | bool | tuple[int, ...] | list[int] | None,
    ) -> tuple[int, ...]:
        """Infer the output shape for the infer_shape operation.

        Args:
            *args (Tensor): Positional args.
            **kwargs (Tensor | int | float | str | bool | tuple[int, ...] | list[int] | None): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        x = args[0] if len(args) > 0 else kwargs.get("x")
        shape = getattr(x, "shape", ())
        return tuple(shape) if shape is not None else ()


@register_op("Pmean")
class Pmean(ReductionOp):
    """Parallel mean reduction operation."""

    op_name = "Pmean"

    def infer_shape(
        self,
        *args: Tensor,
        **kwargs: Tensor | int | float | str | bool | tuple[int, ...] | list[int] | None,
    ) -> tuple[int, ...]:
        """Infer shape.

        Args:
            *args (Tensor): Positional args.
            **kwargs (Tensor | int | float | str | bool | tuple[int, ...] | list[int] | None): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        x = args[0] if len(args) > 0 else kwargs.get("x")
        shape = getattr(x, "shape", ())
        return tuple(shape) if shape is not None else ()
