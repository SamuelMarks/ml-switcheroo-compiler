"""Tensor array ops."""

from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("TensorArrayRead")
class TensorArrayRead(OpDef):
    """Tensor array read."""

    op_name = "TensorArrayRead"

    def infer_shape(self, handle: object, index: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            handle (object): The handle parameter.
            index (object): The index parameter.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
        """
        # Typically returns the shape of the elements in the TensorArray
        return getattr(handle, "element_shape", ())


@register_op("TensorArrayWrite")
class TensorArrayWrite(OpDef):
    """Tensor array write."""

    op_name = "TensorArrayWrite"

    def infer_shape(self, handle: object, index: object, value: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            handle (object): The handle parameter.
            index (object): The index parameter.
            value (object): The value parameter.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
        """
        return ()


@register_op("TensorArrayStack")
class TensorArrayStack(OpDef):
    """Tensor array stack."""

    op_name = "TensorArrayStack"

    def infer_shape(self, handle: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            handle (object): The handle parameter.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
        """
        # Adds a dimension
        elem_shape = getattr(handle, "element_shape", ())
        size = getattr(handle, "size", None)
        if size is not None and isinstance(size, int):
            return (size,) + elem_shape
        return (None,) + elem_shape


__all__ = [
    "TensorArrayRead",
    "TensorArrayStack",
    "TensorArrayWrite",
]
