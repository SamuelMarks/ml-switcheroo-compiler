"""Shape operations for Tensor objects."""

from __future__ import annotations

from typing import TYPE_CHECKING
from ml_switcheroo_compiler.ops.base import dispatch_eager

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.errors import UnimplementedMathError
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

if TYPE_CHECKING:
    from collections.abc import Sequence


@dispatch_eager("Reshape")
def reshape(input: Tensor, shape: Sequence[int]) -> Tensor:
    """Reshapes the input tensor to the specified shape.

    Args:
        input (Tensor): The input tensor to reshape
        shape (Sequence[int]): The target shape

    Returns:
    Tensor: A reshaped tensor with the specified shape
    """
    inputs = [input]
    # shape calculation placeholder
    out_shape = tuple(shape)
    return _emit_shape_node(
        "Reshape",
        inputs,
        {},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


@dispatch_eager("Flatten")
def flatten(input: Tensor, start_dim: int = 0, end_dim: int = -1) -> Tensor:
    """Flattens the input tensor into a 1D tensor.

    Args:
        input (Tensor): The input tensor to flatten
        start_dim (int): The first dimension to flatten. Defaults to 0
        end_dim (int): The last dimension to flatten. Defaults to -1

    Returns:
    Tensor: A flattened 1D tensor
    """
    inputs = [input]
    # shape calculation placeholder
    out_shape = inputs[0].shape
    return _emit_shape_node(
        "Flatten",
        inputs,
        {},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


@dispatch_eager("Squeeze")
def squeeze(input: Tensor, dim: int | Sequence[int] | None = None) -> Tensor:
    """Removes dimensions of size 1 from the input tensor.

    Args:
        input (Tensor): The input tensor
        dim (int | Sequence[int] | None): The dimension(s) to squeeze
        If None, all dimensions of size 1 are removed. Defaults to None

    Returns:
    Tensor: The squeezed tensor
    """
    inputs = [input]
    if dim is None:
        out_shape = tuple(s for s in input.shape if s != 1)
    else:
        dims = [dim] if isinstance(dim, int) else dim
        out_shape = tuple(s for i, s in enumerate(input.shape) if i not in dims or s != 1)
    return _emit_shape_node(
        "Squeeze",
        inputs,
        {"dim": dim} if dim is not None else {},
        out_shape,
        input.dtype,
    )


@dispatch_eager("ExpandDims")
def expand_dims(a: Tensor, axis: int | Sequence[int]) -> Tensor:
    """Expands the shape of an array.

    Args:
        a (Tensor): The input tensor
        axis (Union[int, Sequence[int]]): The position(s) in the expanded axes where the new axis
        (or axes) is placed.

    Returns:
    Tensor: The expanded tensor
    """
    # Calculate new shape
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    out_shape = list(a.shape)

    # Sort axes to handle insertions correctly
    axes_sorted = sorted([ax if ax >= 0 else ax + len(out_shape) + 1 for ax in axes])
    for ax in axes_sorted:
        out_shape.insert(ax, 1)

    out_shape_tup = tuple(out_shape)

    return reshape(a, out_shape_tup)


def unsqueeze(input: Tensor, dim: int) -> Tensor:
    """Inserts a dimension of size 1 at the specified position.

    Args:
        input (Tensor): The input.
        dim (int): The dim.

    Returns:
        Tensor: The computed result.
    """
    return expand_dims(input, dim)


@dispatch_eager("BroadcastTo")
def expand(input: Tensor, size: Sequence[int]) -> Tensor:
    """Expands the tensor to a new shape by broadcasting singleton dimensions.

    Args:
        input (Tensor): The input tensor
        size (Sequence[int]): The desired expanded shape

    Returns:
    Tensor: The expanded tensor
    """
    inputs = [input]
    # shape calculation placeholder
    out_shape = inputs[0].shape
    return _emit_shape_node(
        "Expand",
        inputs,
        {},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


@dispatch_eager("BroadcastTo")
def broadcast_to(input: Tensor, size: Sequence[int]) -> Tensor:
    """Broadcasts the input tensor to a new shape.

    Args:
        input (Tensor): The input tensor
        size (Sequence[int]): The target shape to broadcast to

    Returns:
    Tensor: The broadcasted tensor
    """
    inputs = [input]
    # shape calculation placeholder
    out_shape = tuple(size)
    return _emit_shape_node(
        "BroadcastTo",
        inputs,
        {},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


@dispatch_eager("Swapaxes")
def transpose(input: Tensor, dim0: int, dim1: int) -> Tensor:
    """Swaps two dimensions of the input tensor.

    Args:
        input (Tensor): The input tensor
        dim0 (int): The first dimension to swap
        dim1 (int): The second dimension to swap

    Returns:
    Tensor: The transposed tensor
    """
    inputs = [input]
    # shape calculation placeholder
    out_shape = inputs[0].shape
    return _emit_shape_node(
        "Transpose",
        inputs,
        {},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


@dispatch_eager("Transpose")
def permute(input: Tensor, dims: Sequence[int]) -> Tensor:
    """Permutes the dimensions of the input tensor according to the specified order.

    Args:
        input (Tensor): The input tensor
        dims (Sequence[int]): The desired ordering of dimensions

    Returns:
    Tensor: The permuted tensor
    """
    inputs = [input]
    # shape calculation placeholder
    out_shape = tuple(input.shape[d] for d in dims)
    return _emit_shape_node(
        "Permute",
        inputs,
        {},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


@dispatch_eager("Swapaxes")
def swapaxes(input: Tensor, axis1: int, axis2: int) -> Tensor:
    """Swaps two axes of the input tensor.

    Args:
        input (Tensor): The input tensor
        axis1 (int): The first axis to swap
        axis2 (int): The second axis to swap

    Returns:
    Tensor: The tensor with swapped axes
    """
    inputs = [input]
    # shape calculation placeholder
    out_shape = inputs[0].shape
    return _emit_shape_node(
        "Swapaxes",
        inputs,
        {},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


@dispatch_eager("Moveaxis")
def moveaxis(
    input: Tensor,
    source: int | Sequence[int],
    destination: int | Sequence[int],
) -> Tensor:
    """Moves axes of the input tensor from source positions to destination positions.

    Args:
        input (Tensor): The input tensor
        source (int | Sequence[int]): Original positions of the axes to move
        destination (int | Sequence[int]): Target positions for the axes

    Returns:
    Tensor: The tensor with moved axes
    """
    inputs = [input]
    # shape calculation placeholder
    out_shape = inputs[0].shape
    return _emit_shape_node(
        "Moveaxis",
        inputs,
        {},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


@dispatch_eager("Roll")
def roll(
    input: Tensor,
    shifts: int | Sequence[int],
    dims: int | Sequence[int] | None = None,
) -> Tensor:
    """Rolls the tensor along the specified dimensions.

    Args:
        input (Tensor): The input tensor
        shifts (int | Sequence[int]): The number of places by which elements are
        shifted
        dims (int | Sequence[int] | None): The dimensions along which to roll. Defaults
        to None

    Returns:
    Tensor: The rolled tensor
    """
    inputs = [input]
    # shape calculation placeholder
    out_shape = inputs[0].shape
    return _emit_shape_node(
        "Roll",
        inputs,
        {},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


def broadcast_in_dim(
    operand: Tensor,
    shape: Sequence[int],
    broadcast_dimensions: Sequence[int],
) -> Tensor:
    """Broadcasts an array to a target shape by matching specified dimensions.

    Args:
        operand (Tensor): The input tensor
        shape (Sequence[int]): The target shape
        broadcast_dimensions (Sequence[int]): The dimensions of the target shape to match

    Returns:
    Tensor: The broadcasted tensor

    Raises:
    UnimplementedMathError: If called in eager mode
    """
    if config.eager_mode:
        msg = "No direct numpy for broadcast_in_dim"
        raise UnimplementedMathError(msg)

    inputs = [operand]
    attributes = {"shape": shape, "broadcast_dimensions": broadcast_dimensions}
    return _emit_shape_node("BroadcastInDim", inputs, attributes, tuple(shape), operand.dtype)
