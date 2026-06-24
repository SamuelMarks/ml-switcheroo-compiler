"""Shape operations for Tensor objects."""

from __future__ import annotations
# pylint: disable=duplicate-code


from ml_switcheroo_compiler.core.constants import MAGIC_VAL_2

from typing import TYPE_CHECKING

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, dispatch_eager, register_op
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


if TYPE_CHECKING:
    from collections.abc import Sequence


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
        return Tensor(data, TensorConfig(data.shape, input.dtype, input.device))
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
        return Tensor(data, TensorConfig(data.shape, input.dtype, input.device))
    inputs = [input, update]
    # shape calculation placeholder
    out_shape = inputs[0].shape
    return _emit_shape_node(
        "DynamicUpdateSlice",
        inputs,
        {},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


@dispatch_eager("DynamicUpdateSlice")
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
        """Infer shape.

        Args:
            *args (object): x, start_indices, and slice_sizes.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        slice_sizes = args[2] if len(args) > MAGIC_VAL_2 else kwargs["slice_sizes"]
        return tuple(slice_sizes)

    def emit_jax(self, *args: object, **kwargs: object) -> object:
        """Emit jax code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return "Not implemented"

    def emit_keras(self, *args: object, **kwargs: object) -> object:
        """Emit keras code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return "Not implemented"

    def emit_mlx(self, *args: object, **kwargs: object) -> object:
        """Emit mlx code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return "Not implemented"

    def emit_pytorch(self, *args: object, **kwargs: object) -> object:
        """Emit pytorch code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return "Not implemented"

    def emit_tensorflow(self, *args: object, **kwargs: object) -> object:
        """Emit tensorflow code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return "Not implemented"


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
            object: The evaluated output resulting from this operation.
        """
        return getattr(x, "shape", ())

    def emit_jax(self, *args: object, **kwargs: object) -> object:
        """Emit jax code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return "Not implemented"

    def emit_keras(self, *args: object, **kwargs: object) -> object:
        """Emit keras code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return "Not implemented"

    def emit_mlx(self, *args: object, **kwargs: object) -> object:
        """Emit mlx code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return "Not implemented"

    def emit_pytorch(self, *args: object, **kwargs: object) -> object:
        """Emit pytorch code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return "Not implemented"

    def emit_tensorflow(self, *args: object, **kwargs: object) -> object:
        """Emit tensorflow code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return "Not implemented"
