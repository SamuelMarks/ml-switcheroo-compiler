"""Defines shape, memory, and movement operations for Tensor objects.

This module provides functions to manipulate tensor shapes, dimensions, and memory
layouts, supporting both eager execution (using NumPy) and lazy execution (by tracing
and emitting logical nodes to a graph)
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import numpy as np
from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.errors import UnimplementedMathError
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.tracing import ProxyTensor, _tracer

if TYPE_CHECKING:
    from collections.abc import Sequence


def _emit_shape_node(
    op_type: str,
    inputs: Sequence[Tensor],
    attrs: dict,
    out_shape: tuple,
    out_dtype: DType,
) -> Tensor:
    """Emits a logical shape node to the tracer and returns a new Tensor.

    Args:
    op_type (str): The name of the operation to emit
    inputs (Sequence[Tensor]): The input tensors for the operation
    attrs (dict): Attributes associated with the operation
    out_shape (tuple): The expected shape of the output tensor
    out_dtype (DType): The data type of the output tensor

    Returns:
    Tensor: A new Tensor representing the output of the emitted node
    """
    out_id = str(uuid.uuid4())
    node = LogicalNode(
        id=out_id,
        op_type=op_type,
        inputs=[inp.data.id for inp in inputs],
        attributes=attrs,
        shape_metadata=out_shape,
    )
    _tracer.add_node(node)
    proxy = ProxyTensor(id=out_id, shape=out_shape, dtype=out_dtype.value)
    device = inputs[0].device if len(inputs) > 0 else config.default_device
    return Tensor(data=proxy, shape=out_shape, dtype=out_dtype, device=device)


def reshape(input: Tensor, shape: Sequence[int]) -> Tensor:
    """Reshapes the input tensor to the specified shape.

    Args:
    input (Tensor): The input tensor to reshape
    shape (Sequence[int]): The target shape

    Returns:
    Tensor: A reshaped tensor with the specified shape
    """
    if config.eager_mode:
        data = np.reshape(input.data, shape)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
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


def flatten(input: Tensor, start_dim: int = 0, end_dim: int = -1) -> Tensor:
    """Flattens the input tensor into a 1D tensor.

    Args:
    input (Tensor): The input tensor to flatten
    start_dim (int): The first dimension to flatten. Defaults to 0
    end_dim (int): The last dimension to flatten. Defaults to -1

    Returns:
    Tensor: A flattened 1D tensor
    """
    if config.eager_mode:
        data = np.reshape(input.data, -1)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    inputs = [input]
    # shape calculation placeholder
    out_shape = inputs[0].shape if len(inputs) > 0 else ()
    return _emit_shape_node(
        "Flatten",
        inputs,
        {},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


def squeeze(input: Tensor, dim: int | Sequence[int] | None = None) -> Tensor:
    """Removes dimensions of size 1 from the input tensor.

    Args:
    input (Tensor): The input tensor
    dim (int | Sequence[int] | None): The dimension(s) to squeeze
        If None, all dimensions of size 1 are removed. Defaults to None

    Returns:
    Tensor: The squeezed tensor
    """
    if config.eager_mode:
        data = np.squeeze(input.data, axis=dim)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
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


def unsqueeze(input: Tensor, dim: int) -> Tensor:
    """Inserts a dimension of size 1 at the specified position.

    Args:
    input (Tensor): The input tensor
    dim (int): The index at which to insert the singleton dimension

    Returns:
    Tensor: The unsqueezed tensor with an extra dimension
    """
    if config.eager_mode:
        data = np.expand_dims(input.data, axis=dim)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    inputs = [input]
    # shape calculation placeholder
    out_shape = inputs[0].shape if len(inputs) > 0 else ()
    return _emit_shape_node(
        "Unsqueeze",
        inputs,
        {},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


def expand(input: Tensor, size: Sequence[int]) -> Tensor:
    """Expands the tensor to a new shape by broadcasting singleton dimensions.

    Args:
    input (Tensor): The input tensor
    size (Sequence[int]): The desired expanded shape

    Returns:
    Tensor: The expanded tensor
    """
    if config.eager_mode:
        data = np.broadcast_to(input.data, size)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    inputs = [input]
    # shape calculation placeholder
    out_shape = inputs[0].shape if len(inputs) > 0 else ()
    return _emit_shape_node(
        "Expand",
        inputs,
        {},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


def broadcast_to(input: Tensor, size: Sequence[int]) -> Tensor:
    """Broadcasts the input tensor to a new shape.

    Args:
    input (Tensor): The input tensor
    size (Sequence[int]): The target shape to broadcast to

    Returns:
    Tensor: The broadcasted tensor
    """
    if config.eager_mode:
        data = np.broadcast_to(input.data, size)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
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


def transpose(input: Tensor, dim0: int, dim1: int) -> Tensor:
    """Swaps two dimensions of the input tensor.

    Args:
    input (Tensor): The input tensor
    dim0 (int): The first dimension to swap
    dim1 (int): The second dimension to swap

    Returns:
    Tensor: The transposed tensor
    """
    if config.eager_mode:
        data = np.swapaxes(input.data, dim0, dim1)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    inputs = [input]
    # shape calculation placeholder
    out_shape = inputs[0].shape if len(inputs) > 0 else ()
    return _emit_shape_node(
        "Transpose",
        inputs,
        {},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


def permute(input: Tensor, dims: Sequence[int]) -> Tensor:
    """Permutes the dimensions of the input tensor according to the specified order.

    Args:
    input (Tensor): The input tensor
    dims (Sequence[int]): The desired ordering of dimensions

    Returns:
    Tensor: The permuted tensor
    """
    if config.eager_mode:
        data = np.transpose(input.data, dims)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
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


def swapaxes(input: Tensor, axis1: int, axis2: int) -> Tensor:
    """Swaps two axes of the input tensor.

    Args:
    input (Tensor): The input tensor
    axis1 (int): The first axis to swap
    axis2 (int): The second axis to swap

    Returns:
    Tensor: The tensor with swapped axes
    """
    if config.eager_mode:
        data = np.swapaxes(input.data, axis1, axis2)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    inputs = [input]
    # shape calculation placeholder
    out_shape = inputs[0].shape if len(inputs) > 0 else ()
    return _emit_shape_node(
        "Swapaxes",
        inputs,
        {},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


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
    if config.eager_mode:
        data = np.moveaxis(input.data, source, destination)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    inputs = [input]
    # shape calculation placeholder
    out_shape = inputs[0].shape if len(inputs) > 0 else ()
    return _emit_shape_node(
        "Moveaxis",
        inputs,
        {},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


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
    if config.eager_mode:
        data = np.roll(input.data, shifts, axis=dims)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    inputs = [input]
    # shape calculation placeholder
    out_shape = inputs[0].shape if len(inputs) > 0 else ()
    return _emit_shape_node(
        "Roll",
        inputs,
        {},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


def slice(
    input: Tensor,
    dim: int,
    start: int | None = None,
    end: int | None = None,
    step: int = 1,
) -> Tensor:
    """Slices the input tensor along a specific dimension.

    Args:
    input (Tensor): The input tensor
    dim (int): The dimension along which to slice
    start (int | None): The starting index of the slice. Defaults to None
    end (int | None): The ending index of the slice. Defaults to None
    step (int): The step size of the slice. Defaults to 1

    Returns:
    Tensor: The sliced tensor
    """
    if config.eager_mode:
        import builtins

        sl = [builtins.slice(None)] * len(input.shape)
        sl[dim] = builtins.slice(start, end, step)
        data = input.data[tuple(sl)]
        return Tensor(data, data.shape, input.dtype, input.device)
    inputs = [input]
    # shape calculation placeholder
    out_shape = inputs[0].shape if len(inputs) > 0 else ()
    return _emit_shape_node(
        "Slice",
        inputs,
        {},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


def dynamic_slice(
    input: Tensor,
    start_indices: Sequence[Tensor],
    slice_sizes: Sequence[int],
) -> Tensor:
    """Slices the input tensor dynamically using start indices and slice sizes.

    Args:
    input (Tensor): The input tensor
    start_indices (Sequence[Tensor]): Dynamic start indices for each dimension
    slice_sizes (Sequence[int]): The size of the slice for each dimension

    Returns:
    Tensor: The dynamically sliced tensor

    Raises:
    UnimplementedMathError: If called in eager mode
    """
    if config.eager_mode:
        msg = "No direct numpy for dynamic_slice"
        raise UnimplementedMathError(msg)
    inputs = [input]
    # shape calculation placeholder
    out_shape = inputs[0].shape if len(inputs) > 0 else ()
    return _emit_shape_node(
        "DynamicSlice",
        inputs,
        {},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


def update_slice(input: Tensor, update: Tensor, start_indices: Sequence[int]) -> Tensor:
    """Updates a slice of the input tensor with an update tensor at specified start.

    indices

    Args:
    input (Tensor): The input tensor to be updated
    update (Tensor): The tensor containing the update values
    start_indices (Sequence[int]): The starting indices where the update should be
    applied

    Returns:
    Tensor: The updated tensor

    Raises:
    UnimplementedMathError: If called in eager mode
    """
    if config.eager_mode:
        msg = "No direct numpy for update_slice"
        raise UnimplementedMathError(msg)
    inputs = [input, update]
    # shape calculation placeholder
    out_shape = inputs[0].shape if len(inputs) > 0 else ()
    return _emit_shape_node(
        "UpdateSlice",
        inputs,
        {},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


def strided_slice(
    input: Tensor,
    begin: Sequence[int],
    end: Sequence[int],
    strides: Sequence[int],
) -> Tensor:
    """Extracts a strided slice from the input tensor.

    Args:
    input (Tensor): The input tensor
    begin (Sequence[int]): The starting indices of the slice
    end (Sequence[int]): The ending indices of the slice
    strides (Sequence[int]): The step sizes for each dimension

    Returns:
    Tensor: The sliced tensor

    Raises:
    UnimplementedMathError: If called in eager mode
    """
    if config.eager_mode:
        msg = "No direct numpy for strided_slice"
        raise UnimplementedMathError(msg)
    inputs = [input]
    # shape calculation placeholder
    out_shape = inputs[0].shape if len(inputs) > 0 else ()
    return _emit_shape_node(
        "StridedSlice",
        inputs,
        {},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


def concatenate(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    """Concatenates a sequence of tensors along a specified dimension.

    Args:
    tensors (Sequence[Tensor]): The sequence of tensors to concatenate
    dim (int): The dimension along which to concatenate. Defaults to 0

    Returns:
    Tensor: The concatenated tensor
    """
    if config.eager_mode:
        data = np.concatenate([getattr(t, "data", t) for t in tensors], axis=dim)
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
        data = np.stack([t.data for t in tensors], axis=dim)
        return Tensor(data, data.shape, tensors[0].dtype, tensors[0].device)
    inputs = list(tensors)
    # shape calculation placeholder
    out_shape = inputs[0].shape if len(inputs) > 0 else ()
    return _emit_shape_node(
        "Stack",
        inputs,
        {},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


def split(
    input: Tensor,
    split_size_or_sections: int | Sequence[int],
    dim: int = 0,
) -> Sequence[Tensor]:
    """Splits the input tensor into multiple sub-tensors.

    Args:
    input (Tensor): The input tensor to split
    split_size_or_sections (int | Sequence[int]): Size of a single chunk or list of
    sizes for each chunk
    dim (int): The dimension along which to split. Defaults to 0

    Returns:
    Sequence[Tensor]: A sequence of sub-tensors
    """
    if config.eager_mode:
        datas = np.split(input.data, split_size_or_sections, axis=dim)
        return tuple(Tensor(d, d.shape, input.dtype, input.device) for d in datas)
    inputs = [input]
    # shape calculation placeholder
    out_shape = inputs[0].shape if len(inputs) > 0 else ()
    return (
        _emit_shape_node(
            "Split",
            inputs,
            {},
            out_shape,
            inputs[0].dtype if len(inputs) > 0 else DType.Float32,
        ),
    )


def unstack(input: Tensor, dim: int = 0) -> Sequence[Tensor]:
    """Unstacks the input tensor along a specified dimension into a sequence of tensors.

    Args:
    input (Tensor): The input tensor to unstack
    dim (int): The dimension along which to unstack. Defaults to 0

    Returns:
    Sequence[Tensor]: A sequence of unstacked tensors
    """
    if config.eager_mode:
        datas = (
            np.unstack(input.data, axis=dim)
            if hasattr(np, "unstack")
            else np.moveaxis(input.data, dim, 0)
        )
        return tuple(Tensor(d, d.shape, input.dtype, input.device) for d in datas)
    inputs = [input]
    # shape calculation placeholder
    out_shape = inputs[0].shape if len(inputs) > 0 else ()
    return (
        _emit_shape_node(
            "Unstack",
            inputs,
            {},
            out_shape,
            inputs[0].dtype if len(inputs) > 0 else DType.Float32,
        ),
    )


def tile(input: Tensor, reps: Sequence[int]) -> Tensor:
    """Constructs a new tensor by repeating the input tensor the specified number of times.

    Args:
    input (Tensor): The input tensor
    reps (Sequence[int]): The number of repetitions along each dimension

    Returns:
    Tensor: The tiled tensor
    """
    if config.eager_mode:
        data = np.tile(input.data, reps)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    inputs = [input]
    # shape calculation placeholder
    out_shape = inputs[0].shape if len(inputs) > 0 else ()
    return _emit_shape_node(
        "Tile",
        inputs,
        {},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


def repeat(
    input: Tensor,
    repeats: int | Sequence[int],
    dim: int | None = None,
) -> Tensor:
    """Repeats elements of the input tensor along a specified dimension.

    Args:
    input (Tensor): The input tensor
    repeats (int | Sequence[int]): The number of repetitions for each element
    dim (int | None): The dimension along which to repeat. Defaults to None

    Returns:
    Tensor: The tensor with repeated elements
    """
    if config.eager_mode:
        data = np.repeat(input.data, repeats, axis=dim)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    inputs = [input]
    # shape calculation placeholder
    out_shape = inputs[0].shape if len(inputs) > 0 else ()
    return _emit_shape_node(
        "Repeat",
        inputs,
        {},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


def gather(input: Tensor, dim: int, index: Tensor) -> Tensor:
    """Gathers values along an axis specified by dim using index tensor.

    Args:
    input (Tensor): The source tensor
    dim (int): The axis along which to index
    index (Tensor): The indices of elements to gather

    Returns:
    Tensor: The gathered tensor
    """
    if config.eager_mode:
        data = np.take_along_axis(input.data, index.data, axis=dim)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    inputs = [input, index]
    # shape calculation placeholder
    out_shape = inputs[0].shape if len(inputs) > 0 else ()
    return _emit_shape_node(
        "Gather",
        inputs,
        {},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


def gather_nd(input: Tensor, indices: Tensor) -> Tensor:
    """Gathers slices from input tensor using multi-dimensional indices.

    Args:
    input (Tensor): The source tensor
    indices (Tensor): Index tensor of shape where the last dimension contains
    indices into input

    Returns:
    Tensor: The gathered tensor

    Raises:
    UnimplementedMathError: If called in eager mode
    """
    if config.eager_mode:
        msg = "No direct numpy for gather_nd"
        raise UnimplementedMathError(msg)
    inputs = [input, indices]
    # shape calculation placeholder
    out_shape = inputs[0].shape if len(inputs) > 0 else ()
    return _emit_shape_node(
        "GatherNd",
        inputs,
        {},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


def scatter(input: Tensor, dim: int, index: Tensor, src: Tensor) -> Tensor:
    """Scatters values from a source tensor into the input tensor along a specified.

    dimension

    Args:
    input (Tensor): The destination tensor
    dim (int): The axis along which to index
    index (Tensor): The indices of elements to scatter
    src (Tensor): The source tensor containing values to scatter

    Returns:
    Tensor: The updated tensor

    Raises:
    UnimplementedMathError: If called in eager mode
    """
    if config.eager_mode:
        msg = "No direct numpy for scatter"
        raise UnimplementedMathError(msg)
    inputs = [input, index, src]
    # shape calculation placeholder
    out_shape = inputs[0].shape if len(inputs) > 0 else ()
    return _emit_shape_node(
        "Scatter",
        inputs,
        {},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


def scatter_nd(indices: Tensor, updates: Tensor, shape: Sequence[int]) -> Tensor:
    """Scatters updates into a new tensor of specified shape using indices.

    Args:
    indices (Tensor): The index tensor
    updates (Tensor): The updates to scatter
    shape (Sequence[int]): The shape of the output tensor

    Returns:
    Tensor: The output tensor with scattered updates

    Raises:
    UnimplementedMathError: If called in eager mode
    """
    if config.eager_mode:
        msg = "No direct numpy for scatter_nd"
        raise UnimplementedMathError(msg)
    inputs = [indices, updates]
    # shape calculation placeholder
    out_shape = inputs[0].shape if len(inputs) > 0 else ()
    return _emit_shape_node(
        "ScatterNd",
        inputs,
        {},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


def scatter_add(input: Tensor, dim: int, index: Tensor, src: Tensor) -> Tensor:
    """Adds values from a source tensor to the input tensor at specified indices along a.

    dimension

    Args:
    input (Tensor): The destination tensor
    dim (int): The axis along which to index
    index (Tensor): The indices of elements to add
    src (Tensor): The source tensor containing values to add

    Returns:
    Tensor: The updated tensor

    Raises:
    UnimplementedMathError: If called in eager mode
    """
    if config.eager_mode:
        msg = "No direct numpy for scatter_add"
        raise UnimplementedMathError(msg)
    inputs = [input, index, src]
    # shape calculation placeholder
    out_shape = inputs[0].shape if len(inputs) > 0 else ()
    return _emit_shape_node(
        "ScatterAdd",
        inputs,
        {},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


def take(input: Tensor, indices: Tensor) -> Tensor:
    """Takes elements from the input tensor at the specified flat indices.

    Args:
    input (Tensor): The input tensor
    indices (Tensor): The flat indices of elements to take

    Returns:
    Tensor: A 1D tensor containing the selected elements
    """
    if config.eager_mode:
        data = np.take(input.data, indices.data)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    inputs = [input, indices]
    # shape calculation placeholder
    out_shape = inputs[0].shape if len(inputs) > 0 else ()
    return _emit_shape_node(
        "Take",
        inputs,
        {},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


def where(condition: Tensor, input: Tensor, other: Tensor) -> Tensor:
    """Selects elements from input or other based on condition.

    Args:
    condition (Tensor): A boolean tensor where True selects from input and False
    from other
    input (Tensor): The tensor to select from where condition is True
    other (Tensor): The tensor to select from where condition is False

    Returns:
    Tensor: The selected tensor
    """
    if config.eager_mode:
        data = np.where(condition.data, input.data, other.data)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    inputs = [condition, input, other]
    # shape calculation placeholder
    out_shape = inputs[0].shape if len(inputs) > 0 else ()
    return _emit_shape_node(
        "Where",
        inputs,
        {},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


def triu(input: Tensor, diagonal: int = 0) -> Tensor:
    """Returns the upper triangular part of a matrix or batch of matrices.

    Args:
    input (Tensor): The input tensor
    diagonal (int): The diagonal to consider. 0 is the main diagonal, positive
    values are above, and negative values are below. Defaults to 0

    Returns:
    Tensor: The upper triangular tensor
    """
    if config.eager_mode:
        data = np.triu(input.data, k=diagonal)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    inputs = [input]
    # shape calculation placeholder
    out_shape = inputs[0].shape if len(inputs) > 0 else ()
    return _emit_shape_node(
        "Triu",
        inputs,
        {},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


def tril(input: Tensor, diagonal: int = 0) -> Tensor:
    """Returns the lower triangular part of a matrix or batch of matrices.

    Args:
    input (Tensor): The input tensor
    diagonal (int): The diagonal to consider. 0 is the main diagonal, positive
    values are above, and negative values are below. Defaults to 0

    Returns:
    Tensor: The lower triangular tensor
    """
    if config.eager_mode:
        data = np.tril(input.data, k=diagonal)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    inputs = [input]
    # shape calculation placeholder
    out_shape = inputs[0].shape if len(inputs) > 0 else ()
    return _emit_shape_node(
        "Tril",
        inputs,
        {},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


def meshgrid(*tensors: Tensor, indexing: str = "ij") -> Sequence[Tensor]:
    """Creates coordinate grids from coordinate vectors.

    Args:
    *tensors (Tensor): Coordinate vectors
    indexing (str): The indexing mode, either "ij" (matrix) or "xy" (Cartesian)
    Defaults to "ij"

    Returns:
    Sequence[Tensor]: A sequence of coordinate grid tensors
    """
    if config.eager_mode:
        datas = np.meshgrid(*[t.data for t in tensors], indexing=indexing)
        return tuple(Tensor(d, d.shape, tensors[0].dtype, tensors[0].device) for d in datas)
    inputs = list(tensors)
    # shape calculation placeholder
    out_shape = inputs[0].shape if len(inputs) > 0 else ()
    return tuple(
        _emit_shape_node(
            "Meshgrid",
            inputs,
            {},
            out_shape,
            inputs[0].dtype if len(inputs) > 0 else DType.Float32,
        )
        for _ in inputs
    )


def pad(
    array: object,
    pad_width: object,
    mode: str = "constant",
    **kwargs: object,
) -> object:
    """Pads an array with specified widths and values.

    Args:
    array (object): The array to pad
    pad_width (object): Number of values padded to the edges of each axis
    mode (str): The padding mode (e.g., 'constant'). Defaults to "constant"
    **kwargs (object): Additional keyword arguments for the padding mode

    Returns:
    object: The padded array
    """
    import numpy as np

    return np.pad(array, pad_width, mode=mode, **kwargs)


def take_along_axis(arr: object, indices: object, axis: int) -> object:
    """Takes values from the input array along a specified axis using 1D indices.

    Args:
    arr (object): The source array or tensor
    indices (object): The indices to take along the axis
    axis (int): The axis along which to take values

    Returns:
    object: The selected values
    """
    import numpy as np

    return np.take_along_axis(
        (arr.data if hasattr(arr, "device") else arr),
        (indices.data if hasattr(indices, "device") else indices),
        axis=axis,
    )
