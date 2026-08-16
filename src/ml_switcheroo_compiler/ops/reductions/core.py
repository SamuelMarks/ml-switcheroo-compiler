"""Module core.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Reductions."""
from typing import Any

from ml_switcheroo_compiler.ops.base import OpDef


class ReductionOp(OpDef):
    """Define base class for reduction operations.

    Provides common functionality for operations that reduce one or more dimensions
    of an input tensor, such as shape inference, NumPy evaluation, and argument
    formatting
    """

    op_name: str = ""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Universal dispatcher for the operation.

        Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

        return dispatch_op(self.op_type, *args, **kwargs)

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer the output shape of the operation.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return ()  # Symbolic shape inference will handle axis reduction logic

    def _format_args(self, x: str, **kwargs: Any) -> str:
        """Format args.

        Args:
            x (str): The x parameter.
            **kwargs (object): Keyword args.

        Returns:
            str: Result.
        """
        args = [x]
        if "axis" in kwargs and kwargs["axis"] is not None:
            args.append(f"axis={kwargs['axis']}")
        if kwargs.get("keepdims"):
            args.append(f"keepdims={kwargs['keepdims']}")
        return ", ".join(args)
