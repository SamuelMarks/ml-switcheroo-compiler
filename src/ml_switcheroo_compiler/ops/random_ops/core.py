"""Core Random."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.registry import ensure_eager_list, ensure_int
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("Rademacher")
class Rademacher(OpDef):
    """Draw samples from the Rademacher distribution."""

    op_name = "Rademacher"

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer the output shape for the infer_shape operation.

        Args:
        *args (TensorOrScalar): Positional args.
        **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        shape = kwargs.get("shape", args[0] if len(args) > 0 else None)
        s = shape if shape is not None else kwargs.get("size")
        if s is None:
            return ()
        if isinstance(s, int):
            return (s,)
        return tuple(int(ensure_int(d)) for d in ensure_eager_list(s))


def rademacher(*args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """Draw samples from a Rademacher distribution.

    Args:
        *args (TensorOrScalar): Positional args.
        **kwargs (TensorOrScalar): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Rademacher", *args, **kwargs)
