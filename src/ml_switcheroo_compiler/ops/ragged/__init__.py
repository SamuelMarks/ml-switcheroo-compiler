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


__all__ = ["RaggedGather", "RaggedTensorToDense"]
