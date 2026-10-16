"""Shape operations for Tensor Anys."""

from __future__ import annotations

import builtins
from collections.abc import Sequence
from typing import cast

from ml_switcheroo_compiler.backends.registry import EagerTensorProtocol, EagerValue
from ml_switcheroo_compiler.core.config import config

# pylint: disable=duplicate-code
from ml_switcheroo_compiler.core.constants import BINARY_OP_ARITY
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, dispatch_eager, register_op
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


def dynamic_slice(
    input: Tensor,
    start_indices: Sequence[Tensor],
    slice_sizes: Sequence[int],
) -> EagerValue:
    """Slice the input tensor dynamically using start indices and slice sizes.

    Args:
        input (Tensor): The input parameter.
        start_indices (Sequence): The start_indices parameter.
        slice_sizes (Sequence): The slice_sizes parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        starts = []
        for s in start_indices:
            if hasattr(s, "data"):
                starts.append(int(cast(int, s.data)))
            else:
                starts.append(int(s))
        starts = [min(max(0, s), d - sz) for s, d, sz in zip(starts, input.shape, slice_sizes)]
        idx = tuple(builtins.slice(s, s + sz) for s, sz in zip(starts, slice_sizes))
        data = cast(EagerTensorProtocol, input.data)[idx]
        return Tensor(data, TensorConfig(data.shape, input.dtype, input.device))
    inputs = [input, *start_indices]
    if len(input.shape) > 0 and len(slice_sizes) != len(input.shape):
        raise ValueError(f"slice_sizes length ({len(slice_sizes)}) must match operand rank ({len(input.shape)})")
    out_shape = tuple(int(s) for s in slice_sizes)
    return _emit_shape_node(
        "DynamicSlice",
        inputs,
        {"slice_sizes": tuple(slice_sizes)},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


def update_slice(input: Tensor, update: Tensor, start_indices: Sequence[int]) -> EagerValue:
    """Update a slice of the input tensor with an update tensor at specified start.

    Args:
        input (Tensor): The input parameter.
        update (Tensor): The update parameter.
        start_indices (Sequence): The start_indices parameter.

    Returns:
        Tensor: Result.
    """
    from ml_switcheroo_compiler.ops.creation.frontend_basic import array

    starts = [s if isinstance(s, Tensor) else array(s) for s in start_indices]
    return dynamic_update_slice(input, update, starts)


@dispatch_eager("DynamicUpdateSlice")
def dynamic_update_slice(
    operand: Tensor,
    update: Tensor,
    start_indices: Sequence[Tensor],
) -> EagerValue:
    """Update a slice of an array at dynamically computed start indices.

    Args:
        operand (Tensor): The operand parameter.
        update (Tensor): The update parameter.
        start_indices (Sequence): The start_indices parameter.

    Returns:
        Tensor: Result.
    """
    inputs = [operand, update, *start_indices]
    return _emit_shape_node(
        "DynamicUpdateSlice",
        inputs,
        {},
        operand.shape,
        operand.dtype,
    )


@register_op("DynamicSlice")
class DynamicSlice(OpDef):
    """DynamicSlice operation."""

    op_name = "DynamicSlice"

    def infer_shape(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Infer the output shape for the infer_shape operation.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        slice_sizes = args[2] if len(args) > BINARY_OP_ARITY else kwargs["slice_sizes"]
        return tuple(getattr(slice_sizes, "__iter__")())


@register_op("DynamicUpdateSlice")
class DynamicUpdateSlice(OpDef):
    """DynamicUpdateSlice operation."""

    op_name = "DynamicUpdateSlice"

    def infer_shape(
        self,
        x: EagerTensorProtocol,
        update: EagerTensorProtocol,
        start_indices: EagerTensorProtocol,
        **kwargs: EagerValue,
    ) -> EagerValue:
        """Infer shape.

        Args:
            x (Any): The input x tensor.
            update (Any): The update parameter for the operation.
            start_indices (Any): The start_indices parameter for the operation.
            **kwargs: Additional keyword arguments.

        Returns: Tensor: The computed result.
        """
        return getattr(x, "shape", ())
