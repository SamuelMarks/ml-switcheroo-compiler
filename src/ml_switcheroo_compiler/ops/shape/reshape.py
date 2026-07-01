# pylint: disable=duplicate-code

"""Defines shape manipulation operations for the ML Switcheroo framework."""

from ml_switcheroo_compiler.core.constants import MAGIC_VAL_3

from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("Reshape")
class Reshape(OpDef):
    """An operator definition for reshaping a tensor to a new shape."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer the output shape of the operation.

        Args:
            *args (object): The first input tensor and the newshape.
            **kwargs (object): Additional keyword arguments.

        Returns:
            The computed shape or evaluation result.
        """
        newshape = args[1] if len(args) > 1 else kwargs["newshape"]
        return newshape


@register_op("Transpose")
class Transpose(OpDef):
    """An operator definition for transposing the dimensions of a tensor."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer the output shape of the operation.

        Args:
            *args (object): The first input tensor and the axes.
            **kwargs (object): Additional keyword arguments.

        Returns:
            The computed shape or evaluation result.
        """
        x = args[0] if len(args) > 0 else kwargs["x"]
        axes = args[1] if len(args) > 1 else kwargs.get("axes", None)
        if isinstance(x, tuple) and axes is not None:
            return tuple(x[i] for i in axes)
        return None

    def _format_args(self, x: str, axes: object) -> str:
        """Evaluate format args.

        Args:
            x (str): Argument x
            axes (object): Argument axes

        Returns:
            str: The evaluated output resulting from this operation.
        """
        return f"{x}" if axes is None else f"{x}, {axes}"


@register_op("BroadcastTo")
class BroadcastTo(OpDef):
    """An operator definition for broadcasting a tensor to a new shape."""

    def infer_shape(self, x: object, shape: object, **kwargs: object) -> object:
        """Infer the output shape of the operation.

        Args:
            x (object): The first input tensor.
            shape (object): The shape of the tensor.
            **kwargs (object): Additional keyword arguments.

        Returns:
            The computed shape or evaluation result.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        if isinstance(x, tuple) and isinstance(shape, tuple):
            try:
                broadcasted = broadcast_shapes(x, shape)
                if broadcasted != shape:
                    raise ValueError(  # pragma: no cover
                        f"[broadcast_shapes] Shapes {x} and {shape} cannot be broadcast."  # pragma: no cover
                    )  # pragma: no cover
            except ValueError as e:  # pragma: no cover
                # Catch the core shape broadcast error and raise the mlx-style one  # pragma: no cover
                raise ValueError(  # pragma: no cover
                    f"[broadcast_shapes] Shapes {x} and {shape} cannot be broadcast."
                ) from e
        return shape


@register_op("BroadcastInDim")
class BroadcastInDim(OpDef):
    """BroadcastInDim operation."""

    op_name = "BroadcastInDim"

    def infer_shape(
        self,
        x: object,
        shape: object,
        broadcast_dimensions: object,
        **kwargs: object,
    ) -> object:
        """Infer shape.

        Args:
            x (object): The input x tensor.
            shape (object): The target shape.
            broadcast_dimensions (object): The broadcast_dimensions parameter for the operation.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return tuple(shape)

    def emit_jax(self, *args: object, **kwargs: object) -> object:
        """Emit jax code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return "Not implemented BroadcastInDim"

    def emit_keras(self, *args: object, **kwargs: object) -> object:
        """Emit keras code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return "Not implemented BroadcastInDim"

    def emit_mlx(self, *args: object, **kwargs: object) -> object:
        """Emit mlx code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return "Not implemented BroadcastInDim"

    def emit_pytorch(self, *args: object, **kwargs: object) -> object:
        """Emit pytorch code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return "Not implemented BroadcastInDim"

    def emit_tensorflow(self, *args: object, **kwargs: object) -> object:
        """Emit tensorflow code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return "Not implemented BroadcastInDim"


@register_op("Resize")
class Resize(OpDef):
    """Resize operation."""

    op_name = "Resize"

    def infer_shape(
        self,
        image: object,
        shape: object,
        method: object = "bilinear",
        **kwargs: object,
    ) -> object:
        """Infer shape.

        Args:
            image (object): The image parameter for the operation.
            shape (object): The target shape.
            method (object): The method parameter for the operation.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        if not hasattr(image, "shape") or not image.shape:
            return ()
        out_shape = list(image.shape)
        # assuming shape is (new_h, new_w) and image is either (..., H, W, C) or something.
        # Often it's (..., H, W, C) in Keras.
        if len(out_shape) >= MAGIC_VAL_3:
            out_shape[-3] = shape[0]
            out_shape[-2] = shape[1]
        return tuple(out_shape)

    def emit_jax(self, *args: object, **kwargs: object) -> object:
        """Emit jax code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return "Not implemented Resize"

    def emit_keras(self, *args: object, **kwargs: object) -> object:
        """Emit keras code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return "Not implemented Resize"

    def emit_mlx(self, *args: object, **kwargs: object) -> object:
        """Emit mlx code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return "Not implemented Resize"

    def emit_pytorch(self, *args: object, **kwargs: object) -> object:
        """Emit pytorch code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return "Not implemented Resize"

    def emit_tensorflow(self, *args: object, **kwargs: object) -> object:
        """Emit tensorflow code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return "Not implemented Resize"


@register_op("Flatten")
class Flatten(OpDef):
    """Flatten operator definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape for Flatten."""
        return ()


@register_op("Squeeze")
class Squeeze(OpDef):
    """Squeeze operator definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape for Squeeze."""
        return ()


@register_op("Swapaxes")
class Swapaxes(OpDef):
    """Swapaxes operator definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape for Swapaxes."""
        return ()


@register_op("Repeat")
class Repeat(OpDef):
    """Repeat operator definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape for Repeat."""
        return ()


@register_op("Permute")
class Permute(OpDef):
    """Permute operator definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape for Permute."""
        return ()


@register_op("Moveaxis")
class Moveaxis(OpDef):
    """Moveaxis operator definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape for Moveaxis."""
        return ()


@register_op("Roll")
class Roll(OpDef):
    """Roll operator definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape for Roll."""
        return ()


@register_op("Tile")
class Tile(OpDef):
    """Tile operator definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape for Tile."""
        return ()


@register_op("Expand")
class Expand(OpDef):
    """Expand operator definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape for Expand."""
        return ()


@register_op("Meshgrid")
class Meshgrid(OpDef):
    """Meshgrid operator definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape for Meshgrid."""
        return ()


@register_op("Tril")
class Tril(OpDef):
    """Tril operator definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape for Tril."""
        return ()


@register_op("Triu")
class Triu(OpDef):
    """Triu operator definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape for Triu."""
        return ()


@register_op("Delete")
class Delete(OpDef):
    """Return a new array with sub-arrays along an axis deleted."""

    op_name = "Delete"
    np_op_name = "delete"

    def infer_shape(self, arr: object, obj: object, axis: int = None, **kwargs: object) -> object:
        """Infer the output shape."""
        return (None,)


@register_op("Diff")
class Diff(OpDef):
    """Calculate the n-th discrete difference along the given axis."""

    op_name = "Diff"
    np_op_name = "diff"

    def infer_shape(self, a: object, n: int = 1, axis: int = -1, **kwargs: object) -> object:
        """Infer the output shape."""
        return (None,)


@register_op("Digitize")
class Digitize(OpDef):
    """Return the indices of the bins to which each value in input array belongs."""

    op_name = "Digitize"
    np_op_name = "digitize"

    def infer_shape(self, x: object, bins: object, right: bool = False, **kwargs: object) -> object:
        """Infer the output shape."""
        return x.shape if hasattr(x, "shape") else ()
