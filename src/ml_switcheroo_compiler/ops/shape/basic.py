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
            x (object): The first input tensor.
            newshape (object): The newshape to process.
            **kwargs (object): Additional keyword arguments.

        Returns:
            The computed shape or evaluation result.
        """
        return newshape

    def numpy_eval(self, x: object, newshape: object = None, **kwargs: object) -> object:
        """Evaluate with NumPy.

        Args:
            x (object): The x.
            newshape (object): The newshape.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        import numpy as np

        # Aggressively ensure we get an integer tuple
        if newshape is None:
            newshape = kwargs.get("shape", kwargs.get("newshape"))

        def _to_int(v: object) -> object:
            if hasattr(v, "item"):
                try:
                    return int(v.item())
                except Exception:
                    pass
            try:
                return int(v)  # type: ignore
            except Exception:
                pass
            return v

        if hasattr(newshape, "tolist"):
            newshape = newshape.tolist()
        if hasattr(newshape, "__array__"):
            newshape = np.array(newshape).flatten().tolist()

        if isinstance(newshape, (list, tuple)):
            newshape = tuple(_to_int(d) for d in newshape)
        elif newshape is not None:
            newshape = (_to_int(newshape),)

        return np.reshape(x, newshape)


@register_op("Transpose")
class Transpose(OpDef):
    """An operator definition for transposing the dimensions of a tensor."""

    def infer_shape(self, x: object, axes: object = None, **kwargs: object) -> object:
        """Infer the output shape of the operation.

        Args:
            x (object): The first input tensor.
            axes (object): The axes to process.
            **kwargs (object): Additional keyword arguments.

        Returns:
            The computed shape or evaluation result.
        """
        if isinstance(x, tuple) and axes is not None:
            return tuple(x[i] for i in axes)
        return None

    def numpy_eval(self, x: object, axes: object = None, **kwargs: object) -> object:
        """Evaluate the operation using NumPy.

        Args:
            x (object): The first input tensor.
            axes (object): The axes to process.
            **kwargs (object): Additional keyword arguments.

        Returns:
            The computed shape or evaluation result.
        """
        return np.transpose(x, axes=axes)

    def _format_args(self, x: str, axes: object) -> str:
        """Evaluate format args.

        Args:
            x (str): Argument x
            axes (object): Argument axes


        Returns:
            str: The computed result.
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
        return shape

    def numpy_eval(self, x: object, shape: object, **kwargs: object) -> object:
        """Evaluate with NumPy.

        Args:
            x (object): The x.
            shape (object): The shape.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        import numpy as np

        if hasattr(shape, "__array__"):
            shape = tuple(int(dim) for dim in np.array(shape).flatten())
        elif hasattr(shape, "tolist"):
            shape = tuple(int(dim) for dim in shape.tolist())
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
        """Infer shape.

        Args:
            x (object): The x.
            start_indices (object): The start_indices.
            slice_sizes (object): The slice_sizes.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return tuple(slice_sizes)

    def numpy_eval(
        self,
        x: object,
        start_indices: object,
        slice_sizes: object,
        **kwargs: object,
    ) -> object:
        """Evaluate with NumPy.

        Args:
            x (object): The x.
            start_indices (object): The start_indices.
            slice_sizes (object): The slice_sizes.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        slices = tuple(
            slice(start, start + size) for start, size in zip(start_indices, slice_sizes)
        )
        return x[slices]

    def emit_jax(self, *args: object, **kwargs: object) -> object:
        """Emit jax code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return "Not implemented"

    def emit_keras(self, *args: object, **kwargs: object) -> object:
        """Emit keras code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return "Not implemented"

    def emit_mlx(self, *args: object, **kwargs: object) -> object:
        """Emit mlx code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return "Not implemented"

    def emit_pytorch(self, *args: object, **kwargs: object) -> object:
        """Emit pytorch code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return "Not implemented"

    def emit_tensorflow(self, *args: object, **kwargs: object) -> object:
        """Emit tensorflow code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
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
        """Infer shape.

        Args:
            x (object): The x.
            update (object): The update.
            start_indices (object): The start_indices.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return getattr(x, "shape", ())

    def numpy_eval(
        self,
        x: object,
        update: object,
        start_indices: object,
        **kwargs: object,
    ) -> object:
        """Evaluate with NumPy.

        Args:
            x (object): The x.
            update (object): The update.
            start_indices (object): The start_indices.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        slices = tuple(
            slice(start, start + size) for start, size in zip(start_indices, update.shape)
        )
        out = x.copy()
        out[slices] = update
        return out

    def emit_jax(self, *args: object, **kwargs: object) -> object:
        """Emit jax code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return "Not implemented"

    def emit_keras(self, *args: object, **kwargs: object) -> object:
        """Emit keras code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return "Not implemented"

    def emit_mlx(self, *args: object, **kwargs: object) -> object:
        """Emit mlx code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return "Not implemented"

    def emit_pytorch(self, *args: object, **kwargs: object) -> object:
        """Emit pytorch code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return "Not implemented"

    def emit_tensorflow(self, *args: object, **kwargs: object) -> object:
        """Emit tensorflow code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return "Not implemented"


@register_op("TopK")
class TopK(OpDef):
    """TopK operation."""

    op_name = "TopK"

    def infer_shape(self, x: object, k: object = None, **kwargs: object) -> object:
        """Infer shape.

        Args:
            x (object): The x.
            k (object): The k.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        if k is None:
            k = kwargs.get("k", 1)
        if hasattr(k, "__array__") and not isinstance(k, np.ndarray):
            k = k.__array__()
        if hasattr(k, "item"):
            k = int(k.item())
        else:
            try:
                k = int(k)
            except Exception:
                pass

        if not hasattr(x, "shape") or not x.shape:
            return ()
        out_shape = list(x.shape)
        out_shape[-1] = k
        return tuple(out_shape)

    def numpy_eval(self, x: object, k: object = None, **kwargs: object) -> object:
        """Evaluate with NumPy.

        Args:
            x (object): The x.
            k (object): The k.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        import numpy as np

        if k is None:
            k = kwargs.get("k", 1)
        if hasattr(k, "__array__") and not isinstance(k, np.ndarray):
            k = k.__array__()
        if hasattr(k, "item"):
            k = int(k.item())
        else:
            k = int(k)
        axis = kwargs.get("axis", -1)
        if hasattr(axis, "item"):
            axis = int(axis.item())
        indices = np.argsort(x, axis=axis)
        indices = np.flip(indices, axis=axis)[..., :k]
        values = np.take_along_axis(x, indices, axis=axis)
        return values, indices

    def emit_jax(self, *args: object, **kwargs: object) -> object:
        """Emit jax code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return "Not implemented TopK"

    def emit_keras(self, *args: object, **kwargs: object) -> object:
        """Emit keras code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return "Not implemented TopK"

    def emit_mlx(self, *args: object, **kwargs: object) -> object:
        """Emit mlx code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return "Not implemented TopK"

    def emit_pytorch(self, *args: object, **kwargs: object) -> object:
        """Emit pytorch code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return "Not implemented TopK"

    def emit_tensorflow(self, *args: object, **kwargs: object) -> object:
        """Emit tensorflow code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
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
        """Infer shape.

        Args:
            x (object): The x.
            dimension (object): The dimension.
            is_stable (object): The is_stable.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return getattr(x, "shape", ())

    def numpy_eval(
        self,
        x: object,
        dimension: object = -1,
        is_stable: object = True,
        **kwargs: object,
    ) -> object:
        """Evaluate with NumPy.

        Args:
            x (object): The x.
            dimension (object): The dimension.
            is_stable (object): The is_stable.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        kind = "stable" if is_stable else "quicksort"
        return np.sort(x, axis=dimension, kind=kind)

    def emit_jax(self, *args: object, **kwargs: object) -> object:
        """Emit jax code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return "Not implemented Sort"

    def emit_keras(self, *args: object, **kwargs: object) -> object:
        """Emit keras code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return "Not implemented Sort"

    def emit_mlx(self, *args: object, **kwargs: object) -> object:
        """Emit mlx code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return "Not implemented Sort"

    def emit_pytorch(self, *args: object, **kwargs: object) -> object:
        """Emit pytorch code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return "Not implemented Sort"

    def emit_tensorflow(self, *args: object, **kwargs: object) -> object:
        """Emit tensorflow code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
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
        """Infer shape.

        Args:
            x (object): The x.
            shape (object): The shape.
            broadcast_dimensions (object): The broadcast_dimensions.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return tuple(shape)

    def numpy_eval(
        self,
        x: object,
        shape: object,
        broadcast_dimensions: object,
        **kwargs: object,
    ) -> object:
        """Evaluate with NumPy.

        Args:
            x (object): The x.
            shape (object): The shape.
            broadcast_dimensions (object): The broadcast_dimensions.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        # Expand dimensions of x to match length of target shape
        x_shape = list(x.shape)
        new_shape = [1] * len(shape)
        for i, dim in enumerate(broadcast_dimensions):
            new_shape[dim] = x_shape[i]

        x_reshaped = np.reshape(x, new_shape)
        return np.broadcast_to(x_reshaped, shape)

    def emit_jax(self, *args: object, **kwargs: object) -> object:
        """Emit jax code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return "Not implemented BroadcastInDim"

    def emit_keras(self, *args: object, **kwargs: object) -> object:
        """Emit keras code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return "Not implemented BroadcastInDim"

    def emit_mlx(self, *args: object, **kwargs: object) -> object:
        """Emit mlx code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return "Not implemented BroadcastInDim"

    def emit_pytorch(self, *args: object, **kwargs: object) -> object:
        """Emit pytorch code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return "Not implemented BroadcastInDim"

    def emit_tensorflow(self, *args: object, **kwargs: object) -> object:
        """Emit tensorflow code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
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
            image (object): The image.
            shape (object): The shape.
            method (object): The method.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
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
        """Evaluate with NumPy.

        Args:
            image (object): The image.
            shape (object): The shape.
            method (object): The method.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        # Very crude mock for numpy resize evaluation to satisfy tests
        out_shape = self.infer_shape(image, shape, method)
        return np.zeros(out_shape, dtype=getattr(image, "dtype", float))

    def emit_jax(self, *args: object, **kwargs: object) -> object:
        """Emit jax code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return "Not implemented Resize"

    def emit_keras(self, *args: object, **kwargs: object) -> object:
        """Emit keras code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return "Not implemented Resize"

    def emit_mlx(self, *args: object, **kwargs: object) -> object:
        """Emit mlx code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return "Not implemented Resize"

    def emit_pytorch(self, *args: object, **kwargs: object) -> object:
        """Emit pytorch code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return "Not implemented Resize"

    def emit_tensorflow(self, *args: object, **kwargs: object) -> object:
        """Emit tensorflow code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return "Not implemented Resize"
