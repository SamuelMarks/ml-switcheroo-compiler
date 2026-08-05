"""Shape operations for Tensor objects."""

from __future__ import annotations

import builtins
from collections.abc import Sequence

from ml_switcheroo_compiler.core.config import config

# pylint: disable=duplicate-code
from ml_switcheroo_compiler.core.constants import MAGIC_VAL_2
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, dispatch_eager, register_op
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


def dynamic_slice(
    input: Tensor,
    start_indices: Sequence[Tensor],
    slice_sizes: Sequence[int],
) -> Tensor:
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
                starts.append(int(s.data))
            else:
                starts.append(int(s))
        starts = [min(max(0, s), d - sz) for s, d, sz in zip(starts, input.shape, slice_sizes)]
        idx = tuple(builtins.slice(s, s + sz) for s, sz in zip(starts, slice_sizes))
        data = input.data[idx]
        return Tensor(data, TensorConfig(data.shape, input.dtype, input.device))
    inputs = [input, *start_indices]
    # shape calculation placeholder
    out_shape = tuple(slice_sizes)
    return _emit_shape_node(
        "DynamicSlice",
        inputs,
        {"slice_sizes": tuple(slice_sizes)},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


def update_slice(input: Tensor, update: Tensor, start_indices: Sequence[int]) -> Tensor:
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
) -> Tensor:
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

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer the output shape for the infer_shape operation.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
        """
        slice_sizes = args[2] if len(args) > MAGIC_VAL_2 else kwargs["slice_sizes"]
        return tuple(slice_sizes)


@register_op("DynamicUpdateSlice")
class DynamicUpdateSlice(OpDef):
    """DynamicUpdateSlice operation."""

    op_name = "DynamicUpdateSlice"

    def infer_shape(
        self,
        x: object,
        update: object,
        start_indices: object,
        **kwargs: object,
    ) -> object:
        """Infer shape.

        Args:
            x (object): The input x tensor.
            update (object): The update parameter for the operation.
            start_indices (object): The start_indices parameter for the operation.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return getattr(x, "shape", ())
