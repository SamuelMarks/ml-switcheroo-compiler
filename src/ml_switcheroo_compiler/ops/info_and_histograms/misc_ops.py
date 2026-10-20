"""Misc operations."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.core.shape import TypedVariadicCallable
from ml_switcheroo_compiler.ops.base import OpDef, register_op

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.core.tensor import Tensor
    from ml_switcheroo_compiler.ops.dispatcher import DispatchArg


class HasShape(typing.Protocol):
    """Protocol for objects providing a shape attribute."""

    shape: tuple[int, ...]


MiscArg = typing.Union[
    "Tensor",
    HasShape,
    TypedVariadicCallable[typing.Union["Tensor", int, float, str, bool, None]],
    int,
    float,
    str,
    bool,
    tuple[int, ...],
    list[int],
    None,
]


@register_op("Infeed")
class Infeed(OpDef):
    """Read from the infeed queue."""

    op_name = "Infeed"

    def infer_shape(
        self,
        *args: MiscArg,
        **kwargs: MiscArg,
    ) -> tuple[int, ...]:
        """Infer shape.

        Args:
            *args (MiscArg): Positional args.
            **kwargs (MiscArg): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        shape_val = kwargs.get("shape", ())
        return tuple(shape_val) if isinstance(shape_val, (tuple, list)) else ()


@register_op("Vectorize")
class Vectorize(OpDef):
    """Generalized function class."""

    op_name = "Vectorize"

    def infer_shape(
        self,
        *args: MiscArg,
        **kwargs: MiscArg,
    ) -> tuple[int, ...]:
        """Infer shape.

        Args:
            *args (MiscArg): Positional args.
            **kwargs (MiscArg): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return ()


@register_op("AxisIndex")
class AxisIndex(OpDef):
    """AxisIndex operation."""

    op_name = "AxisIndex"

    def infer_shape(
        self,
        *args: MiscArg,
        **kwargs: MiscArg,
    ) -> tuple[int, ...]:
        """Infer shape.

        Args:
            *args (MiscArg): Positional args.
            **kwargs (MiscArg): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        if not args:
            return ()
        shape_val = getattr(args[0], "shape", ())
        if isinstance(shape_val, tuple):
            return shape_val
        return tuple(shape_val)


def infeed(
    *args: MiscArg,
    **kwargs: MiscArg,
) -> Tensor | DispatchArg | tuple[Tensor | DispatchArg, ...]:
    """Read from the infeed queue.

    Args:
        *args (MiscArg): Positional args.
        **kwargs (MiscArg): Keyword args.

    Returns:
        Tensor | DispatchArg | tuple[Tensor | DispatchArg, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Infeed", *args, **kwargs)


def vectorize(
    *args: MiscArg,
    **kwargs: MiscArg,
) -> Tensor | DispatchArg | tuple[Tensor | DispatchArg, ...]:
    """Vectorize a python function.

    Args:
        *args (MiscArg): Positional args.
        **kwargs (MiscArg): Keyword args.

    Returns:
        Tensor | DispatchArg | tuple[Tensor | DispatchArg, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Vectorize", *args, **kwargs)
