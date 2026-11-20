"""Space-to-batch and batch-to-space ops."""

from __future__ import annotations

from ml_switcheroo_compiler.core.shape import (
    OpValue,
    extract_data,
    extract_device,
    extract_dtype,
    extract_shape,
)
from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("SpaceToBatchND")
class SpaceToBatchND(OpDef):
    """Space to Batch ND op."""

    op_name = "SpaceToBatchND"

    def infer_shape(
        self,
        input: OpValue,
        block_shape: OpValue,
        paddings: OpValue,
        **kwargs: OpValue,
    ) -> OpValue:
        """Infer shape.

        Args:
            input (OpValue): Input `input` argument (input).
            block_shape (OpValue): Input `block_shape` argument (block shape).
            paddings (OpValue): Input `paddings` argument (paddings).
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        return ()


@register_op("SpaceToBatch")
class SpaceToBatch(OpDef):
    """Space to Batch op."""

    op_name = "SpaceToBatch"

    def infer_shape(
        self,
        input: OpValue,
        block_size: OpValue,
        paddings: OpValue,
        **kwargs: OpValue,
    ) -> OpValue:
        """Infer shape.

        Args:
            input (OpValue): Input `input` argument (input).
            block_size (OpValue): Input `block_size` argument (block size).
            paddings (OpValue): Input `paddings` argument (paddings).
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        return ()


def space_to_batch(input: OpValue, block_size: OpValue, paddings: OpValue, **kwargs: OpValue) -> OpValue:
    """Space to batch operation.

    Args:
        input (OpValue): Input `input` argument (input).
        block_size (OpValue): Input `block_size` argument (block size).
        paddings (OpValue): Input `paddings` argument (paddings).
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    if config.eager_mode:
        data = get_active_backend().execute_op("SpaceToBatch", extract_data(input), block_size, paddings, **kwargs)
        return Tensor(
            data, TensorConfig(extract_shape(data), extract_dtype(input, "float32"), extract_device(input, None))
        )
    from ml_switcheroo_compiler.ops.base import get_op
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    out_shape = get_op("SpaceToBatch")().dispatch_infer_shape(input, block_size=block_size, paddings=paddings, **kwargs)
    return _emit_shape_node(
        "SpaceToBatch",
        [input],
        {"block_size": block_size, "paddings": paddings, **kwargs},
        out_shape,
        extract_dtype(input, "float32"),
    )


def space_to_batch_nd(input: OpValue, block_shape: OpValue, paddings: OpValue, **kwargs: OpValue) -> OpValue:
    """Space to batch ND operation.

    Args:
        input (OpValue): Input `input` argument (input).
        block_shape (OpValue): Input `block_shape` argument (block shape).
        paddings (OpValue): Input `paddings` argument (paddings).
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    if config.eager_mode:
        data = get_active_backend().execute_op("SpaceToBatchND", extract_data(input), block_shape, paddings, **kwargs)
        return Tensor(
            data, TensorConfig(extract_shape(data), extract_dtype(input, "float32"), extract_device(input, None))
        )
    from ml_switcheroo_compiler.ops.base import get_op
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    out_shape = get_op("SpaceToBatchND")().dispatch_infer_shape(
        input, block_shape=block_shape, paddings=paddings, **kwargs
    )
    return _emit_shape_node(
        "SpaceToBatchND",
        [input],
        {"block_shape": block_shape, "paddings": paddings, **kwargs},
        out_shape,
        extract_dtype(input, "float32"),
    )
