"""Dropout operations."""

from __future__ import annotations
from ml_switcheroo_compiler.ops.base import OpDef, register_op, dispatch_eager, get_op
from ml_switcheroo_compiler.core.tensor import Tensor


@register_op("Dropout")
class Dropout(OpDef):
    """Dropout operation."""

    def infer_shape(self, x: object, **kwargs: object) -> object:
        """Infer shape."""
        return x.shape


@dispatch_eager("Dropout")
def dropout(x: Tensor, rate: float = 0.5, training: bool = False, seed: int = None) -> Tensor:
    """Dropout operation.

    Args:
        x: Input tensor.
        rate: Dropout rate.
        training: Whether to apply dropout.
        seed: Random seed.

    Returns:
        Tensor.
    """
    return get_op("Dropout")()(x, rate=rate, training=training, seed=seed)
