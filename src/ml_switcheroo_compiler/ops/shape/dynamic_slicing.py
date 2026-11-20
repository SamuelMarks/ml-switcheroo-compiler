"""Shape operations for Tensor objects."""

from __future__ import annotations

import builtins
from collections.abc import Sequence

from ml_switcheroo_compiler.backends.registry import (
    adapt_backend_module,
    coerce_int,
    coerce_shape,
    get_active_backend,
)
from ml_switcheroo_compiler.core.config import config

# pylint: disable=duplicate-code
from ml_switcheroo_compiler.core.constants import BINARY_OP_ARITY
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import (
    DataHolderProtocol,
    OpValue,
    extract_shape,
)
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, dispatch_eager, register_op
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


def dynamic_slice(
    input: Tensor,
    start_indices: Sequence[Tensor],
    slice_sizes: Sequence[int],
) -> OpValue:
    """Slice the input tensor dynamically using start indices and slice sizes.

    Args:
        input (Tensor): Input `input` argument (input).
        start_indices (Sequence[Tensor]): Input `start_indices` argument (start indices).
        slice_sizes (Sequence[int]): Input `slice_sizes` argument (slice sizes).

    Returns:
        Tensor: Computed `Tensor` output.

    Raises:
        ValueError: If input arguments, shapes, or configuration values in `dynamic_slice` are invalid.
    """
    if config.eager_mode:
        starts = []
        for s in start_indices:
            raw_s: OpValue = s
            if isinstance(raw_s, DataHolderProtocol):
                starts.append(int(coerce_int(raw_s.data)))
            else:
                starts.append(int(coerce_int(raw_s)))
        in_shape = extract_shape(input)
        starts = [min(max(0, s), d - sz) for s, d, sz in zip(starts, in_shape, slice_sizes)]
        idx = tuple(builtins.slice(s, s + sz) for s, sz in zip(starts, slice_sizes))
        data = adapt_backend_module(get_active_backend()).asarray(input.data)[idx]
        return Tensor(data, TensorConfig(extract_shape(data), input.dtype, input.device))
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


def update_slice(input: Tensor, update: Tensor, start_indices: Sequence[int | Tensor]) -> OpValue:
    """Update a slice of the input tensor with an update tensor at specified start.

    Args:
        input (Tensor): Input `input` argument (input).
        update (Tensor): Input `update` argument (update).
        start_indices (Sequence[int | Tensor]): Input `start_indices` argument (start indices).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    from ml_switcheroo_compiler.ops.creation.api_basic import array

    starts = [s if isinstance(s, Tensor) else array(s) for s in start_indices]
    return dynamic_update_slice(input, update, starts)


@dispatch_eager("DynamicUpdateSlice")
def dynamic_update_slice(
    operand: Tensor,
    update: Tensor,
    start_indices: Sequence[Tensor],
) -> OpValue:
    """Update a slice of an array at dynamically computed start indices.

    Args:
        operand (Tensor): Input `operand` argument (operand).
        update (Tensor): Input `update` argument (update).
        start_indices (Sequence[Tensor]): Input `start_indices` argument (start indices).

    Returns:
        Tensor: Computed `Tensor` output.
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

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer the output shape for the infer_shape operation.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        slice_sizes = args[2] if len(args) > BINARY_OP_ARITY else kwargs["slice_sizes"]
        return tuple(coerce_shape(slice_sizes))


@register_op("DynamicUpdateSlice")
class DynamicUpdateSlice(OpDef):
    """DynamicUpdateSlice operation."""

    op_name = "DynamicUpdateSlice"

    def infer_shape(
        self,
        x: OpValue,
        update: OpValue,
        start_indices: OpValue,
        **kwargs: OpValue,
    ) -> OpValue:
        """Infer shape.

        Args:
            x (OpValue): The input x tensor.
            update (OpValue): The update parameter for the operation.
            start_indices (OpValue): The start_indices parameter for the operation.
            **kwargs: Additional keyword arguments.

        Returns: Tensor: The computed result.
        """
        _typed_ret_1: OpValue = extract_shape(x)
        return _typed_ret_1
