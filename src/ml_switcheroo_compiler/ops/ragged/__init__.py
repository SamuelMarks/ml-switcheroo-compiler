"""Ragged ops."""

from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("RaggedGather")
class RaggedGather(OpDef):
    """RaggedGather operation."""

    op_name = "RaggedGather"

    def infer_shape(self, params: object, indices: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("RaggedTensorToDense")
class RaggedTensorToDense(OpDef):
    """RaggedTensorToDense operation."""

    op_name = "RaggedTensorToDense"

    def infer_shape(self, rt_input: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("RaggedAdd")
class RaggedAdd(OpDef):
    """RaggedAdd operation."""

    op_name = "RaggedAdd"

    def infer_shape(self, a: object, b: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("RaggedMatMul")
class RaggedMatMul(OpDef):
    """RaggedMatMul operation."""

    op_name = "RaggedMatMul"

    def infer_shape(self, a: object, b: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("RaggedDynamicBroadcast")
class RaggedDynamicBroadcast(OpDef):
    """RaggedDynamicBroadcast operation."""

    op_name = "RaggedDynamicBroadcast"

    def infer_shape(self, rt_input: object, shape: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


__all__ = [
    "RaggedAdd",
    "RaggedDynamicBroadcast",
    "RaggedGather",
    "RaggedMatMul",
    "RaggedTensorToDense",
]
