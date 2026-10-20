"""Implementation of base operations and utilities for the ops subsystem (`ml_switcheroo_compiler.ops.unary.base`)."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.ops.base import OpDef

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.core.tensor import Tensor
    from ml_switcheroo_compiler.ir.core import IRNode


class UnaryMathOp(OpDef):
    """Define base class for unary mathematical operations.

    Provides default implementations for shape inference and NumPy evaluation
    for operations that take a single input and apply an element-wise mathematical
    transformation

    Attributes:
        op_name (str): The name of the operation
    """

    op_name: str = ""

    def infer_shape(
        self,
        *shapes: tuple[int, ...] | list[int] | Tensor | IRNode,
        **kwargs: str | int | float | bool | None,
    ) -> tuple[int, ...]:
        """Infer the output shape of the operation.

        Args:
            *shapes (tuple[int, ...] | list[int] | Tensor | IRNode): The input shapes or tensors.
            **kwargs (str | int | float | bool | None): Additional keyword arguments.

        Returns:
            tuple[int, ...]: The computed shape.
        """
        if not shapes:
            return ()
        first = shapes[0]
        return tuple(getattr(first, "shape", getattr(first, "shape_metadata", first if isinstance(first, (list, tuple)) else ())))
