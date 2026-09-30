# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Core abstractions and logic definitions for base.py."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ml_switcheroo_compiler.ops.base import OpDef

if TYPE_CHECKING:
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
