"""Shape manipulation operations."""

import numpy as np
from typing import Any
from ml_switcheroo.ops.base import OpDef, register_op


@register_op("Reshape")
class Reshape(OpDef):
    """Docstring."""

    def infer_shape(self, x: object, newshape: object, **kwargs: object) -> object:
        """Docstring."""
        return newshape

    def numpy_eval(self, x: object, newshape: object, **kwargs: object) -> object:
        """Docstring."""
        return np.reshape(x, newshape)

    def vjp(
        self, cotangent: object, x: object, newshape: object, **kwargs: object
    ) -> tuple[Any, ...]:
        """Docstring."""
        return (f"jnp.reshape({cotangent}, {x}.shape)",)

    def jvp(
        self, tangent: object, x: object, newshape: object, **kwargs: object
    ) -> object:
        """Docstring."""
        return f"jnp.reshape({tangent}, {newshape})"

    def emit_jax(self, x: str, newshape: str, **kwargs: object) -> str:
        """Docstring."""
        return f"jnp.reshape({x}, {newshape})"

    def emit_pytorch(self, x: str, newshape: str, **kwargs: object) -> str:
        """Docstring."""
        return f"torch.reshape({x}, {newshape})"

    def emit_mlx(self, x: str, newshape: str, **kwargs: object) -> str:
        """Docstring."""
        return f"mx.reshape({x}, {newshape})"

    def emit_keras(self, x: str, newshape: str, **kwargs: object) -> str:
        """Docstring."""
        return f"keras.ops.reshape({x}, {newshape})"

    def emit_tensorflow(self, x: str, newshape: str, **kwargs: object) -> str:
        """Docstring."""
        return f"tf.reshape({x}, {newshape})"


@register_op("Transpose")
class Transpose(OpDef):
    """Docstring."""

    def infer_shape(self, x: object, axes: object = None, **kwargs: object) -> object:
        """Docstring."""
        if isinstance(x, tuple) and axes is not None:
            return tuple(x[i] for i in axes)
        return None

    def numpy_eval(self, x: object, axes: object = None, **kwargs: object) -> object:
        """Docstring."""
        return np.transpose(x, axes=axes)

    def vjp(
        self, cotangent: object, x: object, axes: object = None, **kwargs: object
    ) -> tuple[Any, ...]:
        """Docstring."""
        if axes is None:
            return (f"jnp.transpose({cotangent})",)
        else:
            return (f"jnp.transpose({cotangent}, axes=invert_permutation({axes}))",)

    def jvp(
        self, tangent: object, x: object, axes: object = None, **kwargs: object
    ) -> object:
        """Docstring."""
        if axes is None:
            return f"jnp.transpose({tangent})"
        else:
            return f"jnp.transpose({tangent}, axes={axes})"

    def _format_args(self, x: str, axes: object) -> str:
        """Docstring."""
        return f"{x}" if axes is None else f"{x}, {axes}"

    def emit_jax(self, x: str, axes: object = None, **kwargs: object) -> str:
        """Docstring."""
        return f"jnp.transpose({self._format_args(x, axes)})"

    def emit_pytorch(self, x: str, axes: object = None, **kwargs: object) -> str:
        """Docstring."""
        if axes is None:
            # PyTorch transpose expects 2 dims, but permute works generally
            return f"{x}.t()"  # Approximation for 2D. Proper would be permute
        return f"torch.permute({x}, {axes})"

    def emit_mlx(self, x: str, axes: object = None, **kwargs: object) -> str:
        """Docstring."""
        return f"mx.transpose({self._format_args(x, axes)})"

    def emit_keras(self, x: str, axes: object = None, **kwargs: object) -> str:
        """Docstring."""
        return f"keras.ops.transpose({self._format_args(x, axes)})"

    def emit_tensorflow(self, x: str, axes: object = None, **kwargs: object) -> str:
        """Docstring."""
        if axes is None:
            return f"tf.transpose({x})"
        return f"tf.transpose({x}, perm={axes})"


@register_op("BroadcastTo")
class BroadcastTo(OpDef):
    """Docstring."""

    def infer_shape(self, x: object, shape: object, **kwargs: object) -> object:
        """Docstring."""
        return shape

    def numpy_eval(self, x: object, shape: object, **kwargs: object) -> object:
        """Docstring."""
        return np.broadcast_to(x, shape)

    def vjp(
        self, cotangent: object, x: object, shape: object, **kwargs: object
    ) -> tuple[Any, ...]:
        """Docstring."""
        return (f"sum_broadcasted({cotangent}, {x}.shape)",)

    def jvp(
        self, tangent: object, x: object, shape: object, **kwargs: object
    ) -> object:
        """Docstring."""
        return f"jnp.broadcast_to({tangent}, {shape})"

    def emit_jax(self, x: str, shape: str, **kwargs: object) -> str:
        """Docstring."""
        return f"jnp.broadcast_to({x}, {shape})"

    def emit_pytorch(self, x: str, shape: str, **kwargs: object) -> str:
        """Docstring."""
        return f"{x}.expand({shape})"

    def emit_mlx(self, x: str, shape: str, **kwargs: object) -> str:
        """Docstring."""
        return f"mx.broadcast_to({x}, {shape})"

    def emit_keras(self, x: str, shape: str, **kwargs: object) -> str:
        """Docstring."""
        return f"keras.ops.broadcast_to({x}, {shape})"

    def emit_tensorflow(self, x: str, shape: str, **kwargs: object) -> str:
        """Docstring."""
        return f"tf.broadcast_to({x}, {shape})"
