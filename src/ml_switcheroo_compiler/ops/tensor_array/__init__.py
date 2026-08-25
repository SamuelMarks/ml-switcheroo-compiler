# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Tensor array ops."""

from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("TensorArrayRead")
class TensorArrayRead(OpDef):
    """Tensor array read."""

    op_name: object = "TensorArrayRead"

    def infer_shape(self, handle: object, index: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            handle (object): The handle parameter.
            index (object): The index parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        # Typically returns the shape of the elements in the TensorArray
        return getattr(handle, "element_shape", ())


@register_op("TensorArrayWrite")
class TensorArrayWrite(OpDef):
    """Tensor array write."""

    op_name: object = "TensorArrayWrite"

    def infer_shape(self, handle: object, index: object, value: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            handle (object): The handle parameter.
            index (object): The index parameter.
            value (object): The value parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return ()


@register_op("TensorArrayStack")
class TensorArrayStack(OpDef):
    """Tensor array stack."""

    op_name: object = "TensorArrayStack"

    def infer_shape(self, handle: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            handle (object): The handle parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        # Adds a dimension
        elem_shape: object = getattr(handle, "element_shape", ())
        size: object = getattr(handle, "size", None)
        if size is not None and isinstance(size, int):
            return (size,) + elem_shape
        return (None,) + elem_shape


__all__ = [
    "TensorArrayRead",
    "TensorArrayStack",
    "TensorArrayWrite",
]
