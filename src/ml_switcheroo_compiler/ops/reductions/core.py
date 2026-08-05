"""Reductions."""

from __future__ import annotations

from ml_switcheroo_compiler.ops.base import OpDef


class ReductionOp(OpDef):
    """Define base class for reduction operations.

    Provides common functionality for operations that reduce one or more dimensions
    of an input tensor, such as shape inference, NumPy evaluation, and argument
    formatting
    """

    op_name: str = ""

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Universal dispatcher for the operation, handling dim/keepdim aliases.

        Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

        Returns:
        object: Result.
        """
        if "dim" in kwargs and "axis" not in kwargs:
            kwargs["axis"] = kwargs.pop("dim")
        if "keepdim" in kwargs and "keepdims" not in kwargs:
            kwargs["keepdims"] = kwargs.pop("keepdim")
        from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

        return dispatch_op(self.op_type, *args, **kwargs)

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer the output shape of the operation.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
        """
        return ()  # Symbolic shape inference will handle axis reduction logic

    def _format_args(self, x: str, **kwargs: object) -> str:
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
