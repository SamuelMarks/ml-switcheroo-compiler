# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Define special binary operations for the ml_switcheroo_compiler framework, including element-.

wise trigonometric, division, and comparison operations
"""

from typing import Any

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.shape import broadcast_shapes as _bs
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.eager_evaluator import EagerEvaluator


@register_op("Atan2")
class Atan2(OpDef):
    """Provide an operation class for computing the element-wise arc tangent of x/y."""

    def infer_shape(self, *shapes: Any, **kwargs: Any) -> Any:
        """Evaluate infer_shape operation.

        Args:
        *shapes (object): Positional args.
        **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        """Infer the output shape of the operation.

        Args:
            *shapes: The input shapes.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed shape or evaluation result.
        """
        if all(isinstance(s, tuple) for s in shapes):
            return _bs(*shapes)
        return shapes[0] if shapes else ()


@register_op("Divmod")
class Divmod(OpDef):
    """Provide an operation class for computing element-wise quotient and remainder."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call Divmod.

        Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        if config.eager_mode:
            return EagerEvaluator.evaluate("Divmod", *args, **kwargs)

        from ml_switcheroo_compiler.ops.binary import floor_divide, remainder

        return (floor_divide(*args, **kwargs), remainder(*args, **kwargs))

    def infer_shape(self, *shapes: Any, **kwargs: Any) -> Any:
        """Evaluate infer_shape operation.

        Args:
            *shapes (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        """Infer the output shape of the operation.

        Args:
            *shapes: The input shapes.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed shape or evaluation result.
        """
        if all(isinstance(s, tuple) for s in shapes):
            return _bs(*shapes)
        return shapes[0] if shapes else ()


@register_op("Allclose")
class Allclose(OpDef):
    """Provide an operation class for checking if two arrays are element-wise equal within a.

    tolerance
    """

    def infer_shape(self, *shapes: Any, **kwargs: Any) -> Any:
        """Evaluate infer_shape operation.

        Args:
            *shapes (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return ()


@register_op("Isclose")
class Isclose(OpDef):
    """Provide an operation class for checking element-wise equality within a tolerance."""

    def infer_shape(self, *shapes: Any, **kwargs: Any) -> Any:
        """Evaluate infer_shape operation.

        Args:
            *shapes (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        """Infer the output shape of the operation.

        Args:
            *shapes: The input shapes.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed shape or evaluation result.
        """
        if all(isinstance(s, tuple) for s in shapes):
            return _bs(*shapes)
        return shapes[0] if shapes else ()
