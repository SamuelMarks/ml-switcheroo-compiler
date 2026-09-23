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
        """Infer the output shape for the TimeDistributed operation.

        Args:
            x (Any): Input tensor with shape (batch, time, *features).
            **kwargs (Any): Additional keyword arguments, optionally containing 'wrapped_op_name'.

        Returns:
            tuple[int, ...]: Inferred output shape preserving batch and time dimensions.
        """
        x_shape = getattr(x, "shape", ())
        if len(x_shape) < 2:
            return x_shape
        batch, time_dim = x_shape[0], x_shape[1]
        feature_shape = x_shape[2:]
        wrapped_op_name = kwargs.get("wrapped_op_name", None)
        if wrapped_op_name:
            try:
                inner_op = get_op(wrapped_op_name)()

                class _InnerDummy:
                    """Dummy placeholder tensor representation for inner shape deduction."""

                    shape = (batch * time_dim, *feature_shape)
                    dtype = getattr(x, "dtype", None)

                inner_kwargs = {k: v for k, v in kwargs.items() if k != "wrapped_op_name"}
                inner_out_shape = inner_op.infer_shape(_InnerDummy(), **inner_kwargs)
                if inner_out_shape and len(inner_out_shape) >= 1:
                    return (batch, time_dim, *inner_out_shape[1:])
            except Exception:
                pass
        return x_shape


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
