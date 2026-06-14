"""Shape operations for Tensor objects."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.errors import UnimplementedMathError
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

if TYPE_CHECKING:
    from collections.abc import Sequence


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
    out_shape = inputs[0].shape
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
        import builtins

        starts = []
        for s in start_indices:
            if hasattr(s, "data"):
                starts.append(int(s.data))
            else:
                starts.append(int(s))
        starts = [min(max(0, s), d - sz) for s, d, sz in zip(starts, input.shape, slice_sizes)]
        idx = tuple(builtins.slice(s, s + sz) for s, sz in zip(starts, slice_sizes))
        data = input.data[idx]
        return Tensor(data, data.shape, input.dtype, input.device)
    inputs = [input]
    # shape calculation placeholder
    out_shape = inputs[0].shape
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
        import builtins

        starts = []
        for s in start_indices:
            if hasattr(s, "data"):
                starts.append(int(s.data))
            else:
                starts.append(int(s))
        starts = [min(max(0, s), d - sz) for s, d, sz in zip(starts, input.shape, update.shape)]
        idx = tuple(builtins.slice(s, s + sz) for s, sz in zip(starts, update.shape))
        data = input.data.copy()
        data[idx] = update.data
        return Tensor(data, data.shape, input.dtype, input.device)
    inputs = [input, update]
    # shape calculation placeholder
    out_shape = inputs[0].shape
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
        import builtins

        idx = tuple(builtins.slice(b, e, s) for b, e, s in zip(begin, end, strides))
        data = input.data[idx]
        return Tensor(data, data.shape, input.dtype, input.device)
    inputs = [input]
    # shape calculation placeholder
    out_shape = inputs[0].shape
    return _emit_shape_node(
        "StridedSlice",
        inputs,
        {},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


def dynamic_update_slice(
    operand: Tensor,
    update: Tensor,
    start_indices: Sequence[Tensor],
) -> Tensor:
    """Updates a slice of an array at dynamically computed start indices.

    Args:
        operand (Tensor): The input tensor
        update (Tensor): The tensor containing the update values
        start_indices (Sequence[Tensor]): Dynamic start indices for each dimension

    Returns:
    Tensor: The dynamically updated tensor

    Raises:
    UnimplementedMathError: If called in eager mode
    """
    if config.eager_mode:
        msg = "No direct numpy for dynamic_update_slice"
        raise UnimplementedMathError(msg)

    inputs = [operand, update]
    return _emit_shape_node(
        "DynamicUpdateSlice",
        inputs,
        {},
        operand.shape,
        operand.dtype,
    )
