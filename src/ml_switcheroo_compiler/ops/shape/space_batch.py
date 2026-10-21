"""Space-to-batch and batch-to-space ops."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.registry import EagerTensorProtocol
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("SpaceToBatchND")
class SpaceToBatchND(OpDef):
    """Space to Batch ND op."""

    op_name = "SpaceToBatchND"

    def infer_shape(self, input: EagerTensorProtocol, block_shape: EagerTensorProtocol, paddings: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            input (EagerTensorProtocol): The input parameter.
            block_shape (EagerTensorProtocol): The block_shape parameter.
            paddings (EagerTensorProtocol): The paddings parameter.
            **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return ()


@register_op("SpaceToBatch")
class SpaceToBatch(OpDef):
    """Space to Batch op."""

    op_name = "SpaceToBatch"

    def infer_shape(self, input: EagerTensorProtocol, block_size: EagerTensorProtocol, paddings: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            input (EagerTensorProtocol): The input parameter.
            block_size (EagerTensorProtocol): The block_size parameter.
            paddings (EagerTensorProtocol): The paddings parameter.
            **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return ()


def space_to_batch(input: TensorOrScalar, block_size: TensorOrScalar, paddings: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """Space to batch operation.

    Args:
        input (TensorOrScalar): The input parameter.
        block_size (TensorOrScalar): The block_size parameter.
        paddings (TensorOrScalar): The paddings parameter.
        **kwargs (TensorOrScalar): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    if config.eager_mode:
        data = get_active_backend().execute_op("SpaceToBatch", getattr(input, "data", input), block_size, paddings, **kwargs)
        return Tensor(data, TensorConfig(getattr(data, "shape", ()), getattr(input, "dtype", "float32"), getattr(input, "device", None)))
    from ml_switcheroo_compiler.ops.base import get_op
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    out_shape = get_op("SpaceToBatch")().infer_shape(input, block_size=block_size, paddings=paddings, **kwargs)
    return _emit_shape_node("SpaceToBatch", [input], {"block_size": block_size, "paddings": paddings, **kwargs}, out_shape, getattr(input, "dtype", "float32"))


def space_to_batch_nd(input: TensorOrScalar, block_shape: TensorOrScalar, paddings: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """Space to batch ND operation.

    Args:
        input (TensorOrScalar): The input parameter.
        block_shape (TensorOrScalar): The block_shape parameter.
        paddings (TensorOrScalar): The paddings parameter.
        **kwargs (TensorOrScalar): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    if config.eager_mode:
        data = get_active_backend().execute_op("SpaceToBatchND", getattr(input, "data", input), block_shape, paddings, **kwargs)
        return Tensor(data, TensorConfig(getattr(data, "shape", ()), getattr(input, "dtype", "float32"), getattr(input, "device", None)))
    from ml_switcheroo_compiler.ops.base import get_op
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    out_shape = get_op("SpaceToBatchND")().infer_shape(input, block_shape=block_shape, paddings=paddings, **kwargs)
    return _emit_shape_node("SpaceToBatchND", [input], {"block_shape": block_shape, "paddings": paddings, **kwargs}, out_shape, getattr(input, "dtype", "float32"))
