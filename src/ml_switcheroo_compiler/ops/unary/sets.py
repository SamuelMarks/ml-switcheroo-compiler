# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Core abstractions and logic definitions for sets.py."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ml_switcheroo_compiler.ops.base import OpDef, register_op

if TYPE_CHECKING:
    from ml_switcheroo_compiler.core.tensor import Tensor
    from ml_switcheroo_compiler.ir.core import IRNode


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
        if not args:
            return ()
        first = args[0]
        if isinstance(first, (tuple, list)):
            return tuple(first)
        return tuple(getattr(first, "shape", getattr(first, "shape_metadata", ())))


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
        if not args:
            return ()
        first = args[0]
        if isinstance(first, (tuple, list)):
            return tuple(first)
        return tuple(getattr(first, "shape", getattr(first, "shape_metadata", ())))


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
        if not args:
            return ()
        first = args[0]
        if isinstance(first, (tuple, list)):
            return tuple(first)
        return tuple(getattr(first, "shape", getattr(first, "shape_metadata", ())))


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
        if not args:
            return ()
        first = args[0]
        if isinstance(first, (tuple, list)):
            return tuple(first)
        return tuple(getattr(first, "shape", getattr(first, "shape_metadata", ())))


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
        if not args:
            return ()
        first = args[0]
        if isinstance(first, (tuple, list)):
            return tuple(first)
        return tuple(getattr(first, "shape", getattr(first, "shape_metadata", ())))


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
        if not args:
            return ()
        first = args[0]
        if isinstance(first, (tuple, list)):
            return tuple(first)
        return tuple(getattr(first, "shape", getattr(first, "shape_metadata", ())))


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
        if not args:
            return ()
        first = args[0]
        if isinstance(first, (tuple, list)):
            return tuple(first)
        return tuple(getattr(first, "shape", getattr(first, "shape_metadata", ())))
