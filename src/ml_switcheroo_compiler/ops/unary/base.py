# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Core abstractions and logic definitions for base.py."""

from typing import Any

from ml_switcheroo_compiler.ops.base import OpDef


class UnaryMathOp(OpDef):
    """Define base class for unary mathematical operations.

    Provides default implementations for shape inference and NumPy evaluation
    for operations that take a single input and apply an element-wise mathematical
    transformation

    Attributes:
    op_name (str): The name of the operation
    """

    op_name: str = ""

    def infer_shape(self, *shapes: Any, **kwargs: Any) -> Any:
        """Infer the output shape of the operation.

        Args:
            *shapes: The input shapes.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed shape or evaluation result.
        """
        return shapes[0] if shapes else ()
