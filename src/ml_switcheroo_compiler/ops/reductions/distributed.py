"""Module distributed.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Reductions."""

from ml_switcheroo_compiler.ops.base import register_op
from ml_switcheroo_compiler.ops.reductions.core import ReductionOp


@register_op("Psum")
class Psum(ReductionOp):
    """Parallel sum reduction operation."""

    op_name = "Psum"

    def infer_shape(self, *args, **kwargs):
        """Infer the output shape for the infer_shape operation.

        Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        x = args[0] if len(args) > 0 else kwargs.get("x")
        return getattr(x, "shape", ())


@register_op("Pmean")
class Pmean(ReductionOp):
    """Parallel mean reduction operation."""

    op_name = "Pmean"

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        x = args[0] if len(args) > 0 else kwargs.get("x")
        return getattr(x, "shape", ())
