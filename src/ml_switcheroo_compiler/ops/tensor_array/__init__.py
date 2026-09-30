# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Tensor array ops."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.base import OpDef, register_op

if TYPE_CHECKING:
    from ml_switcheroo_compiler.core.tensor_array import TensorArray


@register_op("TensorArrayRead")
class TensorArrayRead(OpDef):
    """Tensor array read."""

    op_name = "TensorArrayRead"

    def infer_shape(
        self,
        handle: Tensor | TensorArray,
        index: Tensor | int,
        **kwargs: Tensor | int | float | str | bool | None,
    ) -> tuple[int, ...]:
        """Infer shape.

        Args:
            handle (Tensor | TensorArray): The handle parameter.
            index (Tensor | int): The index parameter.
            **kwargs (Tensor | int | float | str | bool | None): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        elem_shape = getattr(handle, "element_shape", ())
        return tuple(elem_shape) if elem_shape is not None else ()


@register_op("TensorArrayWrite")
class TensorArrayWrite(OpDef):
    """Tensor array write."""

    op_name = "TensorArrayWrite"

    def infer_shape(
        self,
        handle: Tensor | TensorArray,
        index: Tensor | int,
        value: Tensor,
        **kwargs: Tensor | int | float | str | bool | None,
    ) -> tuple[int, ...]:
        """Infer shape.

        Args:
            handle (Tensor | TensorArray): The handle parameter.
            index (Tensor | int): The index parameter.
            value (Tensor): The value parameter.
            **kwargs (Tensor | int | float | str | bool | None): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return ()


@register_op("TensorArrayStack")
class TensorArrayStack(OpDef):
    """Tensor array stack."""

    op_name = "TensorArrayStack"

    def infer_shape(
        self,
        handle: Tensor | TensorArray,
        **kwargs: Tensor | int | float | str | bool | None,
    ) -> tuple[int, ...]:
        """Infer shape.

        Args:
            handle (Tensor | TensorArray): The handle parameter.
            **kwargs (Tensor | int | float | str | bool | None): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        elem_shape = tuple(getattr(handle, "element_shape", ()))
        size = getattr(handle, "size", None)
        if size is not None and isinstance(size, int):
            return (size,) + elem_shape
        return (None,) + elem_shape


__all__ = [
    "TensorArrayRead",
    "TensorArrayStack",
    "TensorArrayWrite",
]
