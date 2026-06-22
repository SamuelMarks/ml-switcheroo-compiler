"""Sparse ops."""

from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("SparseDenseMatMul")
class SparseDenseMatMul(OpDef):
    """SparseDenseMatMul operation."""

    op_name = "SparseDenseMatMul"

    def infer_shape(self, sp_a: object, b: object, **kwargs: object) -> object:
        """Infer shape."""
        # Typically returns shape based on sp_a dense_shape and b
        return ()


@register_op("SparseAdd")
class SparseAdd(OpDef):
    """SparseAdd operation."""

    op_name = "SparseAdd"

    def infer_shape(self, a: object, b: object, **kwargs: object) -> object:
        """Infer shape."""
        return getattr(a, "dense_shape", ())


@register_op("SparseReduceSum")
class SparseReduceSum(OpDef):
    """SparseReduceSum operation."""

    op_name = "SparseReduceSum"

    def infer_shape(self, sp_input: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


__all__ = ["SparseDenseMatMul", "SparseAdd", "SparseReduceSum"]
