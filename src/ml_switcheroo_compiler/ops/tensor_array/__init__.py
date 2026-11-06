"""Tensor array ops."""

from __future__ import annotations

import typing
from collections.abc import Sequence

from ml_switcheroo_compiler.backends.registry import ensure_shape
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.base import OpDef, register_op

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.core.tensor_array import TensorArray


@typing.runtime_checkable
class _ElementShapeProtocol(typing.Protocol):
    """Protocol for TensorArray handles exposing `element_shape`."""

    element_shape: Sequence[int] | None


@typing.runtime_checkable
class _SizeAttrProtocol(typing.Protocol):
    """Protocol for TensorArray handles exposing `size`."""

    size: int | TensorOrScalar


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
        raw_h: Tensor | TensorArray | _ElementShapeProtocol = handle
        elem_shape = raw_h.element_shape if isinstance(raw_h, _ElementShapeProtocol) else ()
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
        raw_h: Tensor | TensorArray | _ElementShapeProtocol | _SizeAttrProtocol = handle
        raw_elem = raw_h.element_shape if isinstance(raw_h, _ElementShapeProtocol) and raw_h.element_shape is not None else ()
        elem_shape = tuple(raw_elem)
        size = raw_h.size if isinstance(raw_h, _SizeAttrProtocol) else None
        if isinstance(size, int):
            return (size,) + elem_shape
        return ensure_shape((None,) + elem_shape)


__all__ = [
    "TensorArrayRead",
    "TensorArrayStack",
    "TensorArrayWrite",
]
