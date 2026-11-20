"""Define special binary operations for the ml_switcheroo_compiler framework, including element-wise.

trigonometric, division, and comparison operations.
"""

from __future__ import annotations

from ml_switcheroo_compiler.backends.registry import coerce_shape
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.shape import broadcast_shapes as _bs
from ml_switcheroo_compiler.core.shape import extract_shape
from ml_switcheroo_compiler.ops.base import MultiOutputOpDef, OpArgValue, OpDef, register_op
from ml_switcheroo_compiler.ops.eager_evaluator import EagerEvaluator


def _infer_broadcast_shape(shapes: tuple[OpArgValue, ...]) -> tuple[int, ...] | None:
    """Infer broadcasted shape across input shapes or objects.

    Args:
        shapes: Tuple of input shape tuples or tensor-like objects.

    Returns:
        tuple[int, ...] | None: Broadcasted output shape tuple, or `None` if any shape is unknown.
    """
    if not shapes or any(s is None for s in shapes):
        return None
    if all(isinstance(s, tuple) for s in shapes):
        return _bs(*[coerce_shape(s) for s in shapes])

    return extract_shape(shapes[0], coerce_shape(shapes[0]))


@register_op("Atan2")
class Atan2(OpDef):
    """Provide an operation class for computing the element-wise arc tangent of x/y."""

    def infer_shape(self, *shapes: OpArgValue, **kwargs: OpArgValue) -> tuple[int, ...] | None:
        """Infer the output shape of the Atan2 operation.

        Args:
            *shapes: The input shapes.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple[int, ...] | None: The broadcasted output shape.
        """
        del kwargs
        return _infer_broadcast_shape(shapes)


@register_op("Divmod")
class Divmod(MultiOutputOpDef):
    """Provide an operation class for computing element-wise quotient and remainder."""

    def __call__(self, *args: OpArgValue, **kwargs: OpArgValue) -> OpArgValue:
        """Call Divmod.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            OpArgValue: Tuple of ``(quotient, remainder)`` tensors.
        """
        if config.eager_mode:
            return EagerEvaluator.evaluate("Divmod", *args, **kwargs)

        from ml_switcheroo_compiler.ops.binary import floor_divide, remainder

        return (floor_divide(*args, **kwargs), remainder(*args, **kwargs))

    def infer_shape(self, *shapes: OpArgValue, **kwargs: OpArgValue) -> tuple[int, ...] | None:
        """Infer the output shape of the Divmod operation.

        Args:
            *shapes: The input shapes.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple[int, ...] | None: The broadcasted output shape.
        """
        del kwargs
        return _infer_broadcast_shape(shapes)


@register_op("Allclose")
class Allclose(OpDef):
    """Provide an operation class for checking if two arrays are element-wise equal within a tolerance."""

    def infer_shape(self, *shapes: OpArgValue, **kwargs: OpArgValue) -> tuple[int, ...]:
        """Compute the infer shape tensor operation.

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

    def infer_shape(self, *shapes: OpArgValue, **kwargs: OpArgValue) -> tuple[int, ...] | None:
        """Infer the output shape of the Isclose operation.

        Args:
            *shapes: The input shapes.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple[int, ...] | None: The broadcasted output shape.
        """
        del kwargs
        return _infer_broadcast_shape(shapes)
