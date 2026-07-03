"""Reductions."""

from __future__ import annotations

from ml_switcheroo_compiler.ops.base import OpDef


class ReductionOp(OpDef):
    """Base class for reduction operations.

    Provides common functionality for operations that reduce one or more dimensions
    of an input tensor, such as shape inference, NumPy evaluation, and argument
    formatting
    """

    op_name: str = ""

    def infer_shape(
        self,
        x: object,
        axis: object = None,
        keepdims: bool = False,
        **kwargs: object,
    ) -> object:
        """Infer the output shape of the operation.

        Args:
            x (object): The first input tensor.
            axis (object): The axis to process.
            keepdims (bool): The keepdims to process.
            **kwargs (object): Additional keyword arguments.

        Returns:
            The computed shape or evaluation result.
        """
        return ()  # Symbolic shape inference will handle axis reduction logic

    def _format_args(self, x: str, **kwargs: object) -> str:
        """Format args.

        Args:
            x (str): The first input tensor.
            **kwargs (object): Additional keyword arguments.

        Returns:
            str: The evaluated output resulting from this operation.
        """
        args = [x]
        if "axis" in kwargs and kwargs["axis"] is not None:
            args.append(f"axis={kwargs['axis']}")
        if kwargs.get("keepdims"):
            args.append(f"keepdims={kwargs['keepdims']}")
        return ", ".join(args)
