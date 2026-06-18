# pylint: disable=duplicate-code

"""Shape operations for Tensor objects."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node
from ml_switcheroo_compiler.ops.base import dispatch_eager, OpDef, register_op

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


@dispatch_eager("DynamicUpdateSlice")
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
