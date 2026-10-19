"""Reductions."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.base import OpDef

if TYPE_CHECKING:
    pass


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
    ) -> Tensor:
        """Universal dispatcher for the operation.

        Args:
            *args (Tensor): Positional tensor arguments.
            **kwargs (int | float | str | bool | tuple[int, ...] | list[int] | None): Keyword arguments.

        Returns:
            Tensor: Resulting tensor or dispatched output.
        """
        from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

        return cast(Tensor, dispatch_op(self.op_type, *args, **kwargs))

    def infer_shape(
        self,
        *args: Tensor,
        **kwargs: Tensor | int | float | str | bool | tuple[int, ...] | list[int] | None,
    ) -> tuple[int, ...]:
        """Infer the output shape of the reduction operation.

        Args:
            *args (Tensor): Positional tensor arguments.
            **kwargs (Tensor | int | float | str | bool | tuple[int, ...] | list[int] | None): Keyword arguments.

        Returns:
            tuple[int, ...]: Inferred output shape.
        """
        if not args or not hasattr(args[0], "shape"):
            return ()
        in_shape = tuple(int(d) for d in args[0].shape if isinstance(d, int))
        ndim = len(in_shape)
        if ndim == 0:
            return ()
        axis = kwargs.get("axis")
        keepdims = bool(kwargs.get("keepdims", False))
        if axis is None:
            return (1,) * ndim if keepdims else ()
        axes_raw = (axis,) if isinstance(axis, int) else tuple(axis) if isinstance(axis, (tuple, list)) else ()
        norm_axes = {a + ndim if a < 0 else a for a in axes_raw if isinstance(a, int)}
        if keepdims:
            return tuple(1 if i in norm_axes else d for i, d in enumerate(in_shape))
        return tuple(d for i, d in enumerate(in_shape) if i not in norm_axes)

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
