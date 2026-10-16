"""Define special binary operations for the ml_switcheroo_compiler framework, including element-wise trigonometric, division, and comparison operations."""

from __future__ import annotations

from typing import cast

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.shape import broadcast_shapes as _bs
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.base import OpArgValue, OpDef, register_op
from ml_switcheroo_compiler.ops.eager_evaluator import EagerEvaluator


def _infer_broadcast_shape(shapes: tuple[OpArgValue, ...]) -> tuple[int, ...]:
    """Infer broadcasted shape across input shapes or objects.

    Args:
        shapes: Tuple of input shape tuples or tensor-like objects.

    Returns:
        tuple[int, ...]: Broadcasted output shape tuple.
    """
    if all(isinstance(s, tuple) for s in shapes):
        return _bs(*cast("tuple[tuple[int, ...], ...]", shapes))

    return cast("tuple[int, ...]", getattr(shapes[0], "shape", shapes[0]))


@register_op("Atan2")
class Atan2(OpDef):
    """Provide an operation class for computing the element-wise arc tangent of x/y."""

    def infer_shape(self, *shapes: OpArgValue, **kwargs: OpArgValue) -> tuple[int, ...]:
        """Infer the output shape of the Atan2 operation.

        Args:
            *shapes: The input shapes.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple[int, ...]: The broadcasted output shape.
        """
        del kwargs
        return _infer_broadcast_shape(shapes)


@register_op("Divmod")
class Divmod(OpDef):
    """Provide an operation class for computing element-wise quotient and remainder."""

    def __call__(self, *args: OpArgValue, **kwargs: OpArgValue) -> Tensor:
        """Call Divmod.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Tensor: Tuple of (quotient, remainder) tensors.
        """
        if config.eager_mode:
            return cast(Tensor, EagerEvaluator.evaluate("Divmod", *args, **kwargs))

        from ml_switcheroo_compiler.ops.binary import floor_divide, remainder

        return cast(Tensor, (floor_divide(*args, **kwargs), remainder(*args, **kwargs)))

    def infer_shape(self, *shapes: OpArgValue, **kwargs: OpArgValue) -> tuple[int, ...]:
        """Infer the output shape of the Divmod operation.

        Args:
            *shapes: The input shapes.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple[int, ...]: The broadcasted output shape.
        """
        del kwargs
        return _infer_broadcast_shape(shapes)


@register_op("Allclose")
class Allclose(OpDef):
    """Provide an operation class for checking if two arrays are element-wise equal within a tolerance."""

    def infer_shape(self, *shapes: OpArgValue, **kwargs: OpArgValue) -> tuple[int, ...]:
        """Evaluate infer_shape operation.

        Args:
            *shapes: Positional args.
            **kwargs: Keyword args.

        Returns:
            tuple[int, ...]: Scalar empty shape tuple.
        """
        del shapes, kwargs
        return ()


@register_op("Isclose")
class Isclose(OpDef):
    """Provide an operation class for checking element-wise equality within a tolerance."""

    def infer_shape(self, *shapes: OpArgValue, **kwargs: OpArgValue) -> tuple[int, ...]:
        """Infer the output shape of the Isclose operation.

        Args:
            *shapes: The input shapes.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple[int, ...]: The broadcasted output shape.
        """
        del kwargs
        return _infer_broadcast_shape(shapes)
