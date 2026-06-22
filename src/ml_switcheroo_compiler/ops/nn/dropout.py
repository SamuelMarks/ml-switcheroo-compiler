"""Dropout operations."""

from __future__ import annotations
from ml_switcheroo_compiler.ops.base import OpDef, register_op, dispatch_eager, get_op
from ml_switcheroo_compiler.core.tensor import Tensor
from collections.abc import Sequence


@register_op("Dropout")
class Dropout(OpDef):
    """Dropout operation."""

    def infer_shape(self, x: object, **kwargs: object) -> object:
        """Infer shape."""
        return x.shape


@dispatch_eager("Dropout")
def dropout(
    x: Tensor,
    rate: float = 0.5,
    noise_shape: Sequence[int] | None = None,
    training: bool = False,
    seed: int = None,
) -> Tensor:
    """Dropout operation.

    Args:
        x: Input tensor.
        rate: Dropout rate.
        noise_shape: Optional shape of the dropout mask.
        training: Whether to apply dropout.
        seed: Random seed.

    Returns:
        Tensor.
    """
    return get_op("Dropout")()(x, rate=rate, noise_shape=noise_shape, training=training, seed=seed)


@register_op("AlphaDropout")
class AlphaDropout(OpDef):
    """AlphaDropout operation."""

    def infer_shape(self, x: object, **kwargs: object) -> object:
        """Infer shape."""
        return x.shape


@dispatch_eager("AlphaDropout")
def alpha_dropout(
    x: Tensor,
    rate: float = 0.5,
    noise_shape: Sequence[int] | None = None,
    training: bool = False,
    seed: int = None,
) -> Tensor:
    """AlphaDropout operation.

    Args:
        x: Input tensor.
        rate: Dropout rate.
        noise_shape: Optional shape of the dropout mask.
        training: Whether to apply dropout.
        seed: Random seed.

    Returns:
        Tensor.
    """
    return get_op("AlphaDropout")()(
        x, rate=rate, noise_shape=noise_shape, training=training, seed=seed
    )


@register_op("ActivityRegularization")
class ActivityRegularization(OpDef):
    """ActivityRegularization operation."""

    def infer_shape(self, x: object, **kwargs: object) -> object:
        """Infer shape."""
        return x.shape


@dispatch_eager("ActivityRegularization")
def activity_regularization(
    x: Tensor,
    l1: float = 0.0,
    l2: float = 0.0,
) -> Tensor:
    """ActivityRegularization operation.

    Args:
        x: Input tensor.
        l1: L1 regularization factor.
        l2: L2 regularization factor.

    Returns:
        Tensor.
    """
    return get_op("ActivityRegularization")()(x, l1=l1, l2=l2)
