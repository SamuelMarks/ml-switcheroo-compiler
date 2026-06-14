"""Shape operations for Tensor objects."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

if TYPE_CHECKING:
    from collections.abc import Sequence


def concatenate(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    """Concatenates a sequence of tensors along a specified dimension.

    Args:
        tensors (Sequence[Tensor]): The sequence of tensors to concatenate
        dim (int): The dimension along which to concatenate. Defaults to 0

    Returns:
    Tensor: The concatenated tensor
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Concatenate", [getattr(t, "data", t) for t in tensors], axis=dim)
        return Tensor(data, data.shape, tensors[0].dtype, tensors[0].device)
    inputs = list(tensors)
    # shape calculation placeholder
    out_shape = tuple(
        sum(t.shape[i] for t in tensors) if i == dim else tensors[0].shape[i]
        for i in range(len(tensors[0].shape))
    )
    return _emit_shape_node(
        "Concatenate",
        inputs,
        {},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


def stack(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    """Stacks a sequence of tensors along a new dimension.

    Args:
        tensors (Sequence[Tensor]): The sequence of tensors to stack
        dim (int): The index of the new dimension to insert. Defaults to 0

    Returns:
    Tensor: The stacked tensor
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Stack", [t.data for t in tensors], axis=dim)
        return Tensor(data, data.shape, tensors[0].dtype, tensors[0].device)
    inputs = list(tensors)
    # shape calculation placeholder
    out_shape = inputs[0].shape
    return _emit_shape_node(
        "Stack",
        inputs,
        {},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


def vstack(tup: Sequence[Tensor]) -> Tensor:
    """Stack arrays in sequence vertically (row wise).

    Args:
        tup (Sequence[Tensor]): The sequence of tensors to stack

    Returns:
        Tensor: The stacked tensor
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Vstack", [t.data for t in tup])
        return Tensor(data, data.shape, tup[0].dtype, tup[0].device)
    inputs = list(tup)
    out_shape = inputs[0].shape
    return _emit_shape_node("Vstack", inputs, {}, out_shape, inputs[0].dtype)


def hstack(tup: Sequence[Tensor]) -> Tensor:
    """Stack arrays in sequence horizontally (column wise).

    Args:
        tup (Sequence[Tensor]): The sequence of tensors to stack

    Returns:
        Tensor: The stacked tensor
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Hstack", [t.data for t in tup])
        return Tensor(data, data.shape, tup[0].dtype, tup[0].device)
    inputs = list(tup)
    out_shape = inputs[0].shape
    return _emit_shape_node("Hstack", inputs, {}, out_shape, inputs[0].dtype)


def dstack(tup: Sequence[Tensor]) -> Tensor:
    """Stack arrays in sequence depth wise (along third axis).

    Args:
        tup (Sequence[Tensor]): The sequence of tensors to stack

    Returns:
        Tensor: The stacked tensor
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Dstack", [t.data for t in tup])
        return Tensor(data, data.shape, tup[0].dtype, tup[0].device)
    inputs = list(tup)
    out_shape = inputs[0].shape
    return _emit_shape_node("Dstack", inputs, {}, out_shape, inputs[0].dtype)
