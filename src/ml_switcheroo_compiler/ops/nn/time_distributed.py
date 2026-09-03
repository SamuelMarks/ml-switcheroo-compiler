"""Module time_distributed.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Time distributed wrapper operations."""

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.base import OpDef, dispatch_eager, get_op, register_op


@register_op("TimeDistributed")
class TimeDistributed(OpDef):
    """TimeDistributed operation."""

    def infer_shape(self, x, **kwargs):
        """Infer the output shape for the infer_shape operation.

        Args:
        x (Any): The x parameter.
        **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        # Note: True shape inference depends on the wrapped op.
        # This is a placeholder since the IR maps it to an identity or reshapes.
        return x.shape


@dispatch_eager("TimeDistributed")
def time_distributed(
    x: Tensor,
    **kwargs,
):
    """TimeDistributed operation.

    Args:
        x: Input tensor.
        **kwargs: Additional keyword arguments for the wrapped operation, must include 'wrapped_op_name'.

    Returns:
        Tensor.
    """
    # For eager, we flatten the time dimension, apply the op, and unflatten.
    # We map this to TimeDistributed IR node with attributes.
    return get_op("TimeDistributed")()(x, **kwargs)
