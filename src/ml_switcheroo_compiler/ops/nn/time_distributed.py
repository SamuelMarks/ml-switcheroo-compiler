"""Time distributed wrapper operations."""

from __future__ import annotations
from ml_switcheroo_compiler.ops.base import OpDef, register_op, dispatch_eager, get_op
from ml_switcheroo_compiler.core.tensor import Tensor


@register_op("TimeDistributed")
class TimeDistributed(OpDef):
    """TimeDistributed operation."""

    def infer_shape(self, x: object, **kwargs: object) -> object:
        """Infer shape."""
        # Note: True shape inference depends on the wrapped op.
        # This is a placeholder since the IR maps it to an identity or reshapes.
        return x.shape


@dispatch_eager("TimeDistributed")
def time_distributed(
    x: Tensor,
    wrapped_op_name: str,
    *args: object,
    **kwargs: object,
) -> Tensor:
    """TimeDistributed operation.

    Args:
        x: Input tensor.
        wrapped_op_name: Name of the wrapped operation.
        *args: Additional arguments for the wrapped operation.
        **kwargs: Additional keyword arguments for the wrapped operation.

    Returns:
        Tensor.
    """
    # For eager, we flatten the time dimension, apply the op, and unflatten.
    # We map this to TimeDistributed IR node with attributes.
    kwargs["wrapped_op_name"] = wrapped_op_name  # pragma: no cover
    return get_op("TimeDistributed")()(x, *args, **kwargs)  # pragma: no cover
