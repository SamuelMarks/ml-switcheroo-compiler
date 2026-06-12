"""Creation operations."""

import numpy as np
from typing import Any
from ml_switcheroo.ops.base import OpDef, register_op


class CreationOp(OpDef):
    """Base class for tensor creation operations."""

    op_name: str = ""

    def infer_shape(self, shape: object, **kwargs: object) -> object:
        """Docstring."""
        return shape

    def numpy_eval(self, shape: object, **kwargs: object) -> object:
        """Docstring."""
        return getattr(np, self.op_name.lower())(shape, **kwargs)

    def vjp(
        self, cotangent: object, *args: object, **kwargs: object
    ) -> tuple[Any, ...]:
        """Docstring."""
        return ()

    def jvp(self, tangent: object, *args: object, **kwargs: object) -> object:
        """Docstring."""
        return "0"

    def emit_jax(self, shape: str, **kwargs: object) -> str:
        """Docstring."""
        return f"jnp.{self.op_name.lower()}({shape})"

    def emit_pytorch(self, shape: str, **kwargs: object) -> str:
        """Docstring."""
        return f"torch.{self.op_name.lower()}({shape})"

    def emit_mlx(self, shape: str, **kwargs: object) -> str:
        """Docstring."""
        return f"mx.{self.op_name.lower()}({shape})"

    def emit_keras(self, shape: str, **kwargs: object) -> str:
        """Docstring."""
        return f"keras.ops.{self.op_name.lower()}({shape})"

    def emit_tensorflow(self, shape: str, **kwargs: object) -> str:
        """Docstring."""
        return f"tf.{self.op_name.lower()}({shape})"


@register_op("Zeros")
class Zeros(CreationOp):
    """Docstring."""

    op_name = "Zeros"


@register_op("Ones")
class Ones(CreationOp):
    """Docstring."""

    op_name = "Ones"


@register_op("Full")
class Full(CreationOp):
    """Docstring."""

    op_name = "Full"

    def infer_shape(
        self, shape: object, fill_value: object, **kwargs: object
    ) -> object:
        """Docstring."""
        return shape

    def numpy_eval(self, shape: object, fill_value: object, **kwargs: object) -> object:
        """Docstring."""
        return np.full(shape, fill_value, **kwargs)

    def emit_jax(self, shape: str, fill_value: str, **kwargs: object) -> str:
        """Docstring."""
        return f"jnp.full({shape}, {fill_value})"

    def emit_pytorch(self, shape: str, fill_value: str, **kwargs: object) -> str:
        """Docstring."""
        return f"torch.full({shape}, {fill_value})"

    def emit_mlx(self, shape: str, fill_value: str, **kwargs: object) -> str:
        """Docstring."""
        return f"mx.full({shape}, {fill_value})"

    def emit_keras(self, shape: str, fill_value: str, **kwargs: object) -> str:
        """Docstring."""
        return f"keras.ops.full({shape}, {fill_value})"

    def emit_tensorflow(self, shape: str, fill_value: str, **kwargs: object) -> str:
        """Docstring."""
        return f"tf.fill({shape}, {fill_value})"


@register_op("Arange")
class Arange(OpDef):
    """Docstring."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Docstring."""
        return None  # Dynamic shape depending on values

    def numpy_eval(self, *args: object, **kwargs: object) -> object:
        """Docstring."""
        return np.arange(*args, **kwargs)

    def vjp(
        self, cotangent: object, *args: object, **kwargs: object
    ) -> tuple[Any, ...]:
        """Docstring."""
        return ()

    def jvp(self, tangent: object, *args: object, **kwargs: object) -> object:
        """Docstring."""
        return "0"

    def emit_jax(self, *args: str, **kwargs: object) -> str:
        """Docstring."""
        ops = ", ".join(args)
        return f"jnp.arange({ops})"

    def emit_pytorch(self, *args: str, **kwargs: object) -> str:
        """Docstring."""
        ops = ", ".join(args)
        return f"torch.arange({ops})"

    def emit_mlx(self, *args: str, **kwargs: object) -> str:
        """Docstring."""
        ops = ", ".join(args)
        return f"mx.arange({ops})"

    def emit_keras(self, *args: str, **kwargs: object) -> str:
        """Docstring."""
        ops = ", ".join(args)
        return f"keras.ops.arange({ops})"

    def emit_tensorflow(self, *args: str, **kwargs: object) -> str:
        """Docstring."""
        ops = ", ".join(args)
        return f"tf.range({ops})"
