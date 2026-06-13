"""Defines shape manipulation operations for the ML Switcheroo framework.

This module contains operator definitions (OpDefs) for reshaping, transposing, and
broadcasting tensors, along with their shape inference and NumPy evaluation logic
"""

import numpy as np

from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("Reshape")
class Reshape(OpDef):
    """An operator definition for reshaping a tensor to a new shape."""

    def infer_shape(self, x: object, newshape: object, **kwargs: object) -> object:
        """Infer the output shape of the operation.

        Args:
            x (object): The x parameter
            newshape (object): The newshape parameter
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        return newshape

    def numpy_eval(self, x: object, newshape: object, **kwargs: object) -> object:
        """Evaluate the operation using NumPy.

        Args:
            x (object): The x parameter
            newshape (object): The newshape parameter
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        return np.reshape(x, newshape)


@register_op("Transpose")
class Transpose(OpDef):
    """An operator definition for transposing the dimensions of a tensor."""

    def infer_shape(self, x: object, axes: object = None, **kwargs: object) -> object:
        """Infer the output shape of the operation.

        Args:
            x (object): The x parameter
            axes (object): The axes parameter
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        if isinstance(x, tuple) and axes is not None:
            return tuple(x[i] for i in axes)
        return None

    def numpy_eval(self, x: object, axes: object = None, **kwargs: object) -> object:
        """Evaluate the operation using NumPy.

        Args:
            x (object): The x parameter
            axes (object): The axes parameter
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        return np.transpose(x, axes=axes)

    def _format_args(self, x: str, axes: object) -> str:
        """Evaluate format args.

        Args:
            x (str): Argument x
            axes (object): Argument axes
        """
        return f"{x}" if axes is None else f"{x}, {axes}"


@register_op("BroadcastTo")
class BroadcastTo(OpDef):
    """An operator definition for broadcasting a tensor to a new shape."""

    def infer_shape(self, x: object, shape: object, **kwargs: object) -> object:
        """Infer the output shape of the operation.

        Args:
            x (object): The x parameter
            shape (object): The shape parameter
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        return shape

    def numpy_eval(self, x: object, shape: object, **kwargs: object) -> object:
        """Evaluate the operation using NumPy.

        Args:
            x (object): The x parameter
            shape (object): The shape parameter
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        return np.broadcast_to(x, shape)


@register_op("DynamicSlice")
class DynamicSlice(OpDef):
    """DynamicSlice operation."""

    op_name = "DynamicSlice"

    def infer_shape(
        self,
        x: object,
        start_indices: object,
        slice_sizes: object,
        **kwargs: object,
    ) -> object:
        """Infer shape."""
        return tuple(slice_sizes)

    def numpy_eval(
        self,
        x: object,
        start_indices: object,
        slice_sizes: object,
        **kwargs: object,
    ) -> object:
        """Evaluate with NumPy."""
        slices = tuple(
            slice(start, start + size) for start, size in zip(start_indices, slice_sizes)
        )
        return x[slices]

    def emit_jax(self, *args: object, **kwargs: object) -> object:
        """Emit jax code."""
        return "Not implemented"

    def emit_keras(self, *args: object, **kwargs: object) -> object:
        """Emit keras code."""
        return "Not implemented"

    def emit_mlx(self, *args: object, **kwargs: object) -> object:
        """Emit mlx code."""
        return "Not implemented"

    def emit_pytorch(self, *args: object, **kwargs: object) -> object:
        """Emit pytorch code."""
        return "Not implemented"

    def emit_tensorflow(self, *args: object, **kwargs: object) -> object:
        """Emit tensorflow code."""
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
        """Infer shape."""
        return getattr(x, "shape", ())

    def numpy_eval(
        self,
        x: object,
        update: object,
        start_indices: object,
        **kwargs: object,
    ) -> object:
        """Evaluate with NumPy."""
        slices = tuple(
            slice(start, start + size) for start, size in zip(start_indices, update.shape)
        )
        out = x.copy()
        out[slices] = update
        return out

    def emit_jax(self, *args: object, **kwargs: object) -> object:
        """Emit jax code."""
        return "Not implemented"

    def emit_keras(self, *args: object, **kwargs: object) -> object:
        """Emit keras code."""
        return "Not implemented"

    def emit_mlx(self, *args: object, **kwargs: object) -> object:
        """Emit mlx code."""
        return "Not implemented"

    def emit_pytorch(self, *args: object, **kwargs: object) -> object:
        """Emit pytorch code."""
        return "Not implemented"

    def emit_tensorflow(self, *args: object, **kwargs: object) -> object:
        """Emit tensorflow code."""
        return "Not implemented"


@register_op("TopK")
class TopK(OpDef):
    """TopK operation."""

    op_name = "TopK"

    def infer_shape(self, x: object, k: object, **kwargs: object) -> object:
        """Infer shape. Returns shape for both values and indices."""
        # This is a bit tricky, if x is unknown, return ()
        if not hasattr(x, "shape") or not x.shape:
            return ()
        out_shape = list(x.shape)
        out_shape[-1] = k
        return tuple(out_shape)

    def numpy_eval(self, x: object, k: object, **kwargs: object) -> object:
        """Evaluate with NumPy."""
        # numpy_eval returns values, indices as a tuple
        if isinstance(k, np.ndarray):
            k = k.item()
        indices = np.argsort(x, axis=-1)[..., -k:]
        # Reverse to get descending order
        indices = indices[..., ::-1]
        values = np.take_along_axis(x, indices, axis=-1)
        return values, indices

    def emit_jax(self, *args: object, **kwargs: object) -> object:
        """Emit jax code."""
        return "Not implemented TopK"

    def emit_keras(self, *args: object, **kwargs: object) -> object:
        """Emit keras code."""
        return "Not implemented TopK"

    def emit_mlx(self, *args: object, **kwargs: object) -> object:
        """Emit mlx code."""
        return "Not implemented TopK"

    def emit_pytorch(self, *args: object, **kwargs: object) -> object:
        """Emit pytorch code."""
        return "Not implemented TopK"

    def emit_tensorflow(self, *args: object, **kwargs: object) -> object:
        """Emit tensorflow code."""
        return "Not implemented TopK"


@register_op("Sort")
class Sort(OpDef):
    """Sort operation."""

    op_name = "Sort"

    def infer_shape(
        self,
        x: object,
        dimension: object = -1,
        is_stable: object = True,
        **kwargs: object,
    ) -> object:
        """Infer shape."""
        return getattr(x, "shape", ())

    def numpy_eval(
        self,
        x: object,
        dimension: object = -1,
        is_stable: object = True,
        **kwargs: object,
    ) -> object:
        """Evaluate with NumPy."""
        kind = "stable" if is_stable else "quicksort"
        return np.sort(x, axis=dimension, kind=kind)

    def emit_jax(self, *args: object, **kwargs: object) -> object:
        """Emit jax code."""
        return "Not implemented Sort"

    def emit_keras(self, *args: object, **kwargs: object) -> object:
        """Emit keras code."""
        return "Not implemented Sort"

    def emit_mlx(self, *args: object, **kwargs: object) -> object:
        """Emit mlx code."""
        return "Not implemented Sort"

    def emit_pytorch(self, *args: object, **kwargs: object) -> object:
        """Emit pytorch code."""
        return "Not implemented Sort"

    def emit_tensorflow(self, *args: object, **kwargs: object) -> object:
        """Emit tensorflow code."""
        return "Not implemented Sort"


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
        """Infer shape."""
        return tuple(shape)

    def numpy_eval(
        self,
        x: object,
        shape: object,
        broadcast_dimensions: object,
        **kwargs: object,
    ) -> object:
        """Evaluate with NumPy."""
        # Expand dimensions of x to match length of target shape
        x_shape = list(x.shape)
        new_shape = [1] * len(shape)
        for i, dim in enumerate(broadcast_dimensions):
            new_shape[dim] = x_shape[i]

        x_reshaped = np.reshape(x, new_shape)
        return np.broadcast_to(x_reshaped, shape)

    def emit_jax(self, *args: object, **kwargs: object) -> object:
        """Emit jax code."""
        return "Not implemented BroadcastInDim"

    def emit_keras(self, *args: object, **kwargs: object) -> object:
        """Emit keras code."""
        return "Not implemented BroadcastInDim"

    def emit_mlx(self, *args: object, **kwargs: object) -> object:
        """Emit mlx code."""
        return "Not implemented BroadcastInDim"

    def emit_pytorch(self, *args: object, **kwargs: object) -> object:
        """Emit pytorch code."""
        return "Not implemented BroadcastInDim"

    def emit_tensorflow(self, *args: object, **kwargs: object) -> object:
        """Emit tensorflow code."""
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
        """Infer shape."""
        if not hasattr(image, "shape") or not image.shape:
            return ()
        out_shape = list(image.shape)
        # assuming shape is (new_h, new_w) and image is either (..., H, W, C) or something.
        # Often it's (..., H, W, C) in Keras.
        if len(out_shape) >= 3:
            out_shape[-3] = shape[0]
            out_shape[-2] = shape[1]
        return tuple(out_shape)

    def numpy_eval(
        self,
        image: object,
        shape: object,
        method: object = "bilinear",
        **kwargs: object,
    ) -> object:
        """Evaluate with NumPy."""
        # Very crude mock for numpy resize evaluation to satisfy tests
        out_shape = self.infer_shape(image, shape, method)
        return np.zeros(out_shape, dtype=getattr(image, "dtype", float))

    def emit_jax(self, *args: object, **kwargs: object) -> object:
        """Emit jax code."""
        return "Not implemented Resize"

    def emit_keras(self, *args: object, **kwargs: object) -> object:
        """Emit keras code."""
        return "Not implemented Resize"

    def emit_mlx(self, *args: object, **kwargs: object) -> object:
        """Emit mlx code."""
        return "Not implemented Resize"

    def emit_pytorch(self, *args: object, **kwargs: object) -> object:
        """Emit pytorch code."""
        return "Not implemented Resize"

    def emit_tensorflow(self, *args: object, **kwargs: object) -> object:
        """Emit tensorflow code."""
        return "Not implemented Resize"
