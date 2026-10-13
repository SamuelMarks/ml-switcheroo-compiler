"""Module core.py."""

from __future__ import annotations

# ruff: noqa: E402, E501, F403

"""Reductions."""

from typing import TYPE_CHECKING, Union

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.base import OpDef

if TYPE_CHECKING:
    from ml_switcheroo_compiler.ops.dispatcher import DispatchArg


class ReductionOp(OpDef):
    """Define base class for reduction operations.

    Provides common functionality for operations that reduce one or more dimensions
    of an input tensor, such as shape inference, NumPy evaluation, and argument
    formatting
    """

    op_name: str = ""

    def __call__(
        self,
        *args: Tensor,
        **kwargs: int | float | str | bool | tuple[int, ...] | list[int] | None,
    ) -> Union[Tensor, DispatchArg, tuple[Union[Tensor, DispatchArg], ...]]:
        """Universal dispatcher for the operation.

        Args:
            *args (Tensor): Positional tensor arguments.
            **kwargs (int | float | str | bool | tuple[int, ...] | list[int] | None): Keyword arguments.

        Returns:
            Tensor | DispatchArg | tuple[Tensor | DispatchArg, ...]: Resulting tensor or dispatched output.
        """
        from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

        return dispatch_op(self.op_type, *args, **kwargs)

    def infer_shape(
        self,
        *args: Tensor,
        **kwargs: Tensor | int | float | str | bool | tuple[int, ...] | list[int] | None,
    ) -> tuple[int, ...]:
        """Infer the output shape of the operation.

        Args:
            *args (Tensor): Positional tensor arguments.
            **kwargs (Tensor | int | float | str | bool | tuple[int, ...] | list[int] | None): Keyword arguments.

        Returns:
            tuple[int, ...]: Inferred output shape.
        """
        return ()  # Symbolic shape inference will handle axis reduction logic

    def _format_args(
        self,
        x: str,
        **kwargs: int | float | str | bool | tuple[int, ...] | list[int] | None,
    ) -> str:
        """Format args.

        Args:
            x (str): The x parameter.
            **kwargs (int | float | str | bool | tuple[int, ...] | list[int] | None): Keyword arguments.

        Returns:
            str: Result.
        """
        args = [x]
        if "axis" in kwargs and kwargs["axis"] is not None:
            args.append(f"axis={kwargs['axis']}")
        if kwargs.get("keepdims"):
            args.append(f"keepdims={kwargs['keepdims']}")
        return ", ".join(args)
