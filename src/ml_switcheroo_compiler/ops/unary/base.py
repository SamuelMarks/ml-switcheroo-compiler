"""Core abstractions and logic definitions for base.py."""

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

    def infer_shape(self, *shapes: object, **kwargs: object) -> object:
        """Infer the output shape of the operation.

        Args:
            *shapes: The input shapes.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed shape or evaluation result.
        """
        return shapes[0] if shapes else ()
