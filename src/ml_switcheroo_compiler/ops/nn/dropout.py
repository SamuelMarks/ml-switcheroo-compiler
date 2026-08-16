"""Module dropout.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Dropout operations."""
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, dispatch_eager, get_op, register_op
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


@dataclass
class DropoutConfig:
    """Configuration parameters for controlling dropout behavior during training.

    Args:
        noise_shape: Optional tuple specifying the shape of the dropout mask.
        training: Boolean flag indicating if the model is in training mode.
        seed: Optional integer seed for random number generation.
    """

    noise_shape: Sequence[int] | None = None
    training: bool = False
    seed: int | None = None


@register_op("Dropout")
class Dropout(OpDef):
    """Operation definition for standard dropout regularization."""

    def infer_shape(self, x: Any, **kwargs: Any) -> Any:
        """Determine the output shape of the dropout operation based on the input.

        Args:
            x: The input Any, typically a tensor, from which to extract the shape.
            **kwargs: Additional keyword arguments.

        Returns:
            The inferred shape, which matches the input shape.
        """
        return x.shape


@dispatch_eager("Dropout")
def dropout(
    x: Tensor,  # type: ignore
    rate: float = 0.5,
    config: DropoutConfig | None = None,
) -> Any:
    """Apply standard dropout regularization to the input tensor.

    Randomly zeroes out elements of the input tensor with the specified probability
    to prevent overfitting during training.

    Args:
        x: The tensor to apply dropout to.
        rate: The probability of zeroing out an element.
        config: Optional configuration object specifying behavior.

    Returns:
        A new tensor with dropout applied.
    """
    return get_op("Dropout")()(x, rate=rate, config=config)


@register_op("AlphaDropout")
class AlphaDropout(OpDef):
    """Operation definition for alpha dropout regularization."""

    def infer_shape(self, x: Any, **kwargs: Any) -> Any:
        """Determine the output shape for alpha dropout based on the input.

        Args:
            x: The input Any, typically a tensor, from which to extract the shape.
            **kwargs: Additional keyword arguments.

        Returns:
            The inferred shape, which matches the input shape.
        """
        return x.shape


@dispatch_eager("AlphaDropout")
def alpha_dropout(
    x: Tensor,  # type: ignore
    rate: float = 0.5,
    config: DropoutConfig | None = None,
) -> Any:
    """Apply alpha dropout regularization to the input tensor.

    Alpha dropout is a type of dropout that maintains the mean and variance
    of the input data, often used with SELU activations.

    Args:
        x: The tensor to apply alpha dropout to.
        rate: The probability of dropping an element.
        config: Optional configuration specifying behavior.

    Returns:
        A new tensor with alpha dropout applied.
    """
    return get_op("AlphaDropout")()(x, rate=rate, config=config)


@register_op("ActivityRegularization")
class ActivityRegularization(OpDef):
    """Operation definition for applying activity regularization."""

    def infer_shape(self, x: Any, **kwargs: Any) -> Any:
        """Determine the output shape for activity regularization based on the input.

        Args:
            x: The input Any, which may be a tensor or a tuple/list of tensors.
            **kwargs: Additional keyword arguments.

        Returns:
            The inferred shape, which matches the input shape or remains unchanged for tuples.
        """
        if isinstance(x, (tuple, list)):
            return x
        return getattr(x, "shape", ())


@dispatch_eager("ActivityRegularization")
def activity_regularization(
    x: Tensor,  # type: ignore
    l1: float = 0.0,
    l2: float = 0.0,
) -> Any:
    """Apply L1 and L2 activity regularization to the given input tensor.

    Adds regularization penalties based on the magnitude of the tensor's values.

    Args:
        x: The input tensor to regularize.
        l1: The L1 regularization factor to apply.
        l2: The L2 regularization factor to apply.

    Returns:
        The unchanged input tensor, as regularization is typically applied as a side effect.
    """
    return get_op("ActivityRegularization")()(x, l1=l1, l2=l2)


@register_op("Dropout1d")
class Dropout1d(OpDef):
    """Operation definition for 1D spatial dropout."""

    op_name = "Dropout1d"

    def infer_shape(self, x: Any, **kwargs: Any) -> Any:
        """Determine the output shape of the 1D dropout operation.

        Args:
            x: The input Any, typically a tensor.
            **kwargs: Additional keyword arguments.

        Returns:
            The inferred shape, matching the input.
        """
        return getattr(x, "shape", ())


def dropout1d(x: Tensor, p: float = 0.5, training: bool = True) -> Any:  # type: ignore
    """Apply 1D spatial dropout to the input tensor.

    Randomly zeroes out entire channels (1D feature maps) of the input tensor.

    Args:
        x: The input tensor to apply 1D dropout to.
        p: The probability of zeroing out a channel.
        training: Boolean flag indicating if the model is in training mode.

    Returns:
        A new tensor with 1D dropout applied.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Dropout1d", x.data, p=p, training=training)
        return Tensor(data, TensorConfig(x.shape, x.dtype, x.device))  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
    return _emit_shape_node("Dropout1d", [x], {"p": p, "training": training}, getattr(x, "shape", ()), getattr(x, "dtype", None))


@register_op("Dropout2d")
class Dropout2d(OpDef):
    """Operation definition for 2D spatial dropout."""

    op_name = "Dropout2d"

    def infer_shape(self, x: Any, **kwargs: Any) -> Any:
        """Determine the output shape of the 2D dropout operation.

        Args:
            x: The input Any, typically a tensor.
            **kwargs: Additional keyword arguments.

        Returns:
            The inferred shape, matching the input shape.
        """
        return getattr(x, "shape", ())


def dropout2d(x: Tensor, p: float = 0.5, training: bool = True) -> Any:  # type: ignore
    """Apply 2D spatial dropout to the input tensor.

    Randomly zeroes out entire 2D feature maps (channels) of the input tensor.

    Args:
        x: The input tensor to apply 2D dropout to.
        p: The probability of zeroing out a channel.
        training: Boolean flag indicating if the model is in training mode.

    Returns:
        A new tensor with 2D dropout applied.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Dropout2d", getattr(x, "data", x), p=p, training=training)
        return Tensor(data, TensorConfig(getattr(data, "shape", ()), getattr(x, "dtype", None), getattr(x, "device", None)))  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    return _emit_shape_node(
        "Dropout2d",
        [x],
        {"p": p, "training": training},
        getattr(x, "shape", ()),
        getattr(x, "dtype", DType.Float32),
    )


@register_op("Dropout3d")
class Dropout3d(OpDef):
    """Operation definition for 3D spatial dropout."""

    op_name = "Dropout3d"

    def infer_shape(self, x: Any, **kwargs: Any) -> Any:
        """Determine the output shape of the 3D dropout operation.

        Args:
            x: The input Any, typically a tensor.
            **kwargs: Additional keyword arguments.

        Returns:
            The inferred shape, matching the input shape.
        """
        return getattr(x, "shape", ())


def dropout3d(x: Tensor, p: float = 0.5, training: bool = True) -> Any:  # type: ignore
    """Apply 3D spatial dropout to the input tensor.

    Randomly zeroes out entire 3D feature maps (channels) of the input tensor.

    Args:
        x: The input tensor to apply 3D dropout to.
        p: The probability of zeroing out a channel.
        training: Boolean flag indicating if the model is in training mode.

    Returns:
        A new tensor with 3D dropout applied.
    """
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
