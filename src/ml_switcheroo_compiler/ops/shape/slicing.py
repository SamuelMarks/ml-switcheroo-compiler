"""Shape operations for Tensor objects."""

from __future__ import annotations

import builtins
from collections.abc import Sequence

# pylint: disable=duplicate-code
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


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
        sl = [builtins.slice(None)] * len(input.shape)
        sl[dim] = builtins.slice(start, end, step)
        data = input.data[tuple(sl)]
        return Tensor(data, TensorConfig(data.shape, input.dtype, input.device))
    inputs = [input]
    # shape calculation placeholder
    out_shape = inputs[0].shape
    return _emit_shape_node(
        "Slice",
        inputs,
        {"dim": dim, "start": start, "end": end, "step": step},
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

    """
    if config.eager_mode:
        idx = tuple(builtins.slice(b, e, s) for b, e, s in zip(begin, end, strides))
        data = input.data[idx]
        return Tensor(data, TensorConfig(data.shape, input.dtype, input.device))
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


@register_op("Slice")
class Slice(OpDef):
    """Slice operator definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape for Slice."""
        return ()


@register_op("StridedSlice")
class StridedSlice(OpDef):
    """StridedSlice operator definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape for StridedSlice."""
        return ()


@register_op("Choose")
class Choose(OpDef):
    """Construct an array from an index array and a list of arrays to choose from."""

    op_name = "Choose"
    np_op_name = "choose"

    def infer_shape(self, a: object, choices: object, out: object = None, mode: str = "raise", **kwargs: object) -> object:
        """Infer the output shape."""
        return a.shape if hasattr(a, "shape") else ()


@register_op("IndexInDim")
class IndexInDim(OpDef):
    """Return elements of an array at specific indices along a given dimension."""

    op_name = "IndexInDim"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        operand = args[0] if len(args) > 0 else None
        index = args[1] if len(args) > 1 else None
        axis = kwargs.get("axis", 0)
        keepdims = kwargs.get("keepdims", True)
        shape = list(getattr(operand, "shape", ()))
        if not shape:
            return ()

        index_shape = getattr(index, "shape", ())
        if keepdims:
            shape[axis] = index_shape[0] if index_shape else 1  # type: ignore[index]
        else:
            if index_shape:
                shape.pop(axis)
                shape = shape[:axis] + list(index_shape) + shape[axis:]
            else:
                shape.pop(axis)

        return tuple(shape)


@register_op("UpdateSlice")
class UpdateSlice(OpDef):
    """Update a slice of an array."""

    op_name = "UpdateSlice"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        operand = args[0] if len(args) > 0 else None
        return getattr(operand, "shape", ())


def index_in_dim(*args: object, **kwargs: object) -> object:
    """Returns the index in a dimension."""
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("IndexInDim", *args, **kwargs)


def update_slice(*args: object, **kwargs: object) -> object:
    """Update a slice."""
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("UpdateSlice", *args, **kwargs)
