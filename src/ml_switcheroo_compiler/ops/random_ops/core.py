"""Core Random."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.registry import coerce_int, coerce_operand_list
from ml_switcheroo_compiler.core.shape import OpValue
from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("Rademacher")
class Rademacher(OpDef):
    """Draw samples from the Rademacher distribution."""

    op_name = "Rademacher"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer the output shape for the infer_shape operation.

        Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        shape = kwargs.get("shape", args[0] if len(args) > 0 else None)
        s = shape if shape is not None else kwargs.get("size")
        if s is None:
            return ()
        if isinstance(s, int):
            return (s,)
        return tuple(int(coerce_int(d)) for d in coerce_operand_list(s))


def rademacher(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Draw samples from a Rademacher distribution.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Rademacher", *args, **kwargs)
