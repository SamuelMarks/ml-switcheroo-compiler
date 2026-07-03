"""Dropout operations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.base import OpDef, dispatch_eager, get_op, register_op
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


@dataclass
class DropoutConfig:
    """Dropout configuration."""

    noise_shape: Sequence[int] | None = None
    training: bool = False
    seed: int | None = None


@register_op("Dropout")
class Dropout(OpDef):
    """Dropout operation."""

    def infer_shape(self, x: object, **kwargs: object) -> object:
        """Infer shape."""
        """Infer shape."""
        return x.shape


@dispatch_eager("Dropout")
def dropout(
    x: Tensor,
    rate: float = 0.5,
    config: DropoutConfig | None = None,
) -> Tensor:
    """Dropout operation.

    Args:
        x: Input tensor.
        rate: Dropout rate.
        config: Dropout configuration.

    Returns:
        Tensor.
    """
    return get_op("Dropout")()(x, rate=rate, config=config)  # pragma: no cover


@register_op("AlphaDropout")
class AlphaDropout(OpDef):
    """AlphaDropout operation."""

    def infer_shape(self, x: object, **kwargs: object) -> object:
        """Infer shape."""
        """Infer shape."""
        return x.shape


@dispatch_eager("AlphaDropout")
def alpha_dropout(
    x: Tensor,
    rate: float = 0.5,
    config: DropoutConfig | None = None,
) -> Tensor:
    """AlphaDropout operation.

    Args:
        x: Input tensor.
        rate: Dropout rate.
        config: Dropout configuration.

    Returns:
        Tensor.
    """
    return get_op("AlphaDropout")()(  # pragma: no cover
        x, rate=rate, config=config
    )


@register_op("ActivityRegularization")
class ActivityRegularization(OpDef):
    """ActivityRegularization operation."""

    def infer_shape(self, x: object, **kwargs: object) -> object:
        """Infer shape."""
        """Infer shape."""
        if isinstance(x, (tuple, list)):
            return x
        return getattr(x, "shape", ())


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
    return get_op("ActivityRegularization")()(x, l1=l1, l2=l2)  # pragma: no cover


@register_op("Dropout2d")
class Dropout2d(OpDef):
    """Dropout2d operation."""

    op_name = "Dropout2d"

    def infer_shape(self, x: object, **kwargs: object) -> object:
        """Infer shape."""
        return getattr(x, "shape", ())


def dropout2d(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    """Dropout2d."""
    if config.eager_mode:
        backend = get_active_backend()
        return backend.execute_op("Dropout2d", getattr(x, "data", x), p=p, training=training)

    return _emit_shape_node(
        "Dropout2d",
        [x],
        {"p": p, "training": training},
        getattr(x, "shape", ()),
        getattr(x, "dtype", DType.Float32),
    )


@register_op("Dropout3d")
class Dropout3d(OpDef):
    """Dropout3d operation."""

    op_name = "Dropout3d"

    def infer_shape(self, x: object, **kwargs: object) -> object:
        """Infer shape."""
        return getattr(x, "shape", ())


def dropout3d(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    """Dropout3d."""
    if config.eager_mode:
        backend = get_active_backend()
        return backend.execute_op("Dropout3d", getattr(x, "data", x), p=p, training=training)

    return _emit_shape_node(
        "Dropout3d",
        [x],
        {"p": p, "training": training},
        getattr(x, "shape", ()),
        getattr(x, "dtype", DType.Float32),
    )
