from __future__ import annotations

# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Shape operations for Tensor objects."""
import builtins
from collections.abc import Sequence
from typing import Any

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
) -> Any:
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
                starts.append(int(s.data))  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
            else:
                starts.append(int(s))
        starts = [min(max(0, s), d - sz) for s, d, sz in zip(starts, input.shape, slice_sizes)]
        idx = tuple(builtins.slice(s, s + sz) for s, sz in zip(starts, slice_sizes))
        data = input.data[idx]  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
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


def update_slice(input: Tensor, update: Tensor, start_indices: Sequence[int]) -> Any:
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
) -> Any:
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

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer the output shape for the infer_shape operation.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        slice_sizes = args[2] if len(args) > MAGIC_VAL_2 else kwargs["slice_sizes"]
        return tuple(slice_sizes)


@register_op("DynamicUpdateSlice")
class DynamicUpdateSlice(OpDef):
    """DynamicUpdateSlice operation."""

    op_name = "DynamicUpdateSlice"

    def infer_shape(
        self,
        x: Any,
        update: Any,
        start_indices: Any,
        **kwargs: Any,
    ) -> Any:
        """Infer shape.

        Args:
            x (object): The input x tensor.
            update (object): The update parameter for the operation.
            start_indices (object): The start_indices parameter for the operation.
            **kwargs: Additional keyword arguments.

        Returns: Any: The computed result.
        """
        return getattr(x, "shape", ())
