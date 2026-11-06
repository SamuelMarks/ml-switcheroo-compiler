"""Implementation of sets operations and utilities for the ops subsystem (`ml_switcheroo_compiler.ops.unary.sets`)."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.core.shape import extract_shape
from ml_switcheroo_compiler.ops.base import OpDef, register_op

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.core.tensor import Tensor
    from ml_switcheroo_compiler.ir.core import IRNode


def _infer_1d_set_shape(args: tuple[tuple[int, ...] | list[int] | Tensor | IRNode, ...]) -> tuple[int, ...]:
    """Extract output shape from the first argument of a 1D set operation.

    Args:
        args (tuple[tuple[int, ...] | list[int] | Tensor | IRNode, ...]): Positional arguments.

    Returns:
        tuple[int, ...]: Extracted shape tuple.
    """
    if not args:
        return ()
    first = args[0]
    if isinstance(first, (tuple, list)):
        return tuple(first)
    return extract_shape(first)


@register_op("Setdiff1d")
class Setdiff1d(OpDef):
    """Setdiff1d operator definition."""

    op_name = "Setdiff1d"

    def infer_shape(
        self,
        *args: tuple[int, ...] | list[int] | Tensor | IRNode,
        **kwargs: str | int | float | bool | None,
    ) -> tuple[int, ...]:
        """Infer output shape for set difference.

        Args:
            *args (tuple[int, ...] | list[int] | Tensor | IRNode): Input shapes or tensors.
            **kwargs (str | int | float | bool | None): Keyword arguments.

        Returns:
            tuple[int, ...]: Output shape.
        """
        del kwargs
        return _infer_1d_set_shape(args)


@register_op("Setxor1d")
class Setxor1d(OpDef):
    """Setxor1d operator definition."""

    op_name = "Setxor1d"

    def infer_shape(
        self,
        *args: tuple[int, ...] | list[int] | Tensor | IRNode,
        **kwargs: str | int | float | bool | None,
    ) -> tuple[int, ...]:
        """Infer output shape for set XOR.

        Args:
            *args (tuple[int, ...] | list[int] | Tensor | IRNode): Input shapes or tensors.
            **kwargs (str | int | float | bool | None): Keyword arguments.

        Returns:
            tuple[int, ...]: Output shape.
        """
        del kwargs
        return _infer_1d_set_shape(args)


@register_op("Union1d")
class Union1d(OpDef):
    """Union1d operator definition."""

    op_name = "Union1d"

    def infer_shape(
        self,
        *args: tuple[int, ...] | list[int] | Tensor | IRNode,
        **kwargs: str | int | float | bool | None,
    ) -> tuple[int, ...]:
        """Infer output shape for set union.

        Args:
            *args (tuple[int, ...] | list[int] | Tensor | IRNode): Input shapes or tensors.
            **kwargs (str | int | float | bool | None): Keyword arguments.

        Returns:
            tuple[int, ...]: Output shape.
        """
        del kwargs
        return _infer_1d_set_shape(args)


@register_op("UniqueAll")
class UniqueAll(OpDef):
    """UniqueAll operator definition."""

    op_name = "UniqueAll"

    def infer_shape(
        self,
        *args: tuple[int, ...] | list[int] | Tensor | IRNode,
        **kwargs: str | int | float | bool | None,
    ) -> tuple[int, ...]:
        """Infer output shape for unique all elements.

        Args:
            *args (tuple[int, ...] | list[int] | Tensor | IRNode): Input shapes or tensors.
            **kwargs (str | int | float | bool | None): Keyword arguments.

        Returns:
            tuple[int, ...]: Output shape.
        """
        del kwargs
        return _infer_1d_set_shape(args)


@register_op("UniqueCounts")
class UniqueCounts(OpDef):
    """UniqueCounts operator definition."""

    op_name = "UniqueCounts"

    def infer_shape(
        self,
        *args: tuple[int, ...] | list[int] | Tensor | IRNode,
        **kwargs: str | int | float | bool | None,
    ) -> tuple[int, ...]:
        """Infer output shape for unique counts.

        Args:
            *args (tuple[int, ...] | list[int] | Tensor | IRNode): Input shapes or tensors.
            **kwargs (str | int | float | bool | None): Keyword arguments.

        Returns:
            tuple[int, ...]: Output shape.
        """
        del kwargs
        return _infer_1d_set_shape(args)


@register_op("UniqueInverse")
class UniqueInverse(OpDef):
    """UniqueInverse operator definition."""

    op_name = "UniqueInverse"

    def infer_shape(
        self,
        *args: tuple[int, ...] | list[int] | Tensor | IRNode,
        **kwargs: str | int | float | bool | None,
    ) -> tuple[int, ...]:
        """Infer output shape for unique inverse indices.

        Args:
            *args (tuple[int, ...] | list[int] | Tensor | IRNode): Input shapes or tensors.
            **kwargs (str | int | float | bool | None): Keyword arguments.

        Returns:
            tuple[int, ...]: Output shape.
        """
        del kwargs
        return _infer_1d_set_shape(args)


@register_op("UniqueValues")
class UniqueValues(OpDef):
    """UniqueValues operator definition."""

    op_name = "UniqueValues"

    def infer_shape(
        self,
        *args: tuple[int, ...] | list[int] | Tensor | IRNode,
        **kwargs: str | int | float | bool | None,
    ) -> tuple[int, ...]:
        """Infer output shape for unique values.

        Args:
            *args (tuple[int, ...] | list[int] | Tensor | IRNode): Input shapes or tensors.
            **kwargs (str | int | float | bool | None): Keyword arguments.

        Returns:
            tuple[int, ...]: Output shape.
        """
        del kwargs
        return _infer_1d_set_shape(args)
