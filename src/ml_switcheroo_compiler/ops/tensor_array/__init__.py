"""Tensor array ops."""

from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("TensorArrayRead")
class TensorArrayRead(OpDef):
    """TensorArrayRead operation."""

    op_name = "TensorArrayRead"

    def infer_shape(self, handle: object, index: object, **kwargs: object) -> object:
        """Infer shape."""
        # Typically returns the shape of the elements in the TensorArray
        return getattr(handle, "element_shape", ())


@register_op("TensorArrayWrite")
class TensorArrayWrite(OpDef):
    """TensorArrayWrite operation."""

    op_name = "TensorArrayWrite"

    def infer_shape(self, handle: object, index: object, value: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("TensorArrayStack")
class TensorArrayStack(OpDef):
    """TensorArrayStack operation."""

    op_name = "TensorArrayStack"

    def infer_shape(self, handle: object, **kwargs: object) -> object:
        """Infer shape."""
        # Adds a dimension
        elem_shape = getattr(handle, "element_shape", ())
        size = getattr(handle, "size", None)
        if size is not None and isinstance(size, int):
            return (size,) + elem_shape
        return (None,) + elem_shape


__all__ = ["TensorArrayRead", "TensorArrayWrite", "TensorArrayStack"]
