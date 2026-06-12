"""Linear algebra operations."""

import numpy as np
from typing import Any
from ml_switcheroo.ops.base import OpDef, register_op


@register_op("Matmul")
class Matmul(OpDef):
    """Docstring."""

    def infer_shape(self, a: object, b: object, **kwargs: object) -> object:
        """Docstring."""
        if isinstance(a, tuple) and isinstance(b, tuple):
            if len(a) >= 2 and len(b) >= 2:
                # Basic matmul shape inference for 2D+
                return a[:-1] + b[1:]
        return None

    def numpy_eval(self, a: object, b: object, **kwargs: object) -> object:
        """Docstring."""
        return np.matmul(a, b)

    def vjp(
        self, cotangent: object, a: object, b: object, **kwargs: object
    ) -> tuple[Any, ...]:
        """Docstring."""
        return (
            f"jnp.matmul({cotangent}, jnp.swapaxes({b}, -1, -2))",
            f"jnp.matmul(jnp.swapaxes({a}, -1, -2), {cotangent})",
        )

    def jvp(
        self,
        tangent_a: object,
        tangent_b: object,
        a: object,
        b: object,
        **kwargs: object,
    ) -> object:
        """Docstring."""
        return f"(jnp.matmul({tangent_a}, {b}) + jnp.matmul({a}, {tangent_b}))"

    def emit_jax(self, a: str, b: str, **kwargs: object) -> str:
        """Docstring."""
        return f"jnp.matmul({a}, {b})"

    def emit_pytorch(self, a: str, b: str, **kwargs: object) -> str:
        """Docstring."""
        return f"torch.matmul({a}, {b})"

    def emit_mlx(self, a: str, b: str, **kwargs: object) -> str:
        """Docstring."""
        return f"mx.matmul({a}, {b})"

    def emit_keras(self, a: str, b: str, **kwargs: object) -> str:
        """Docstring."""
        return f"keras.ops.matmul({a}, {b})"

    def emit_tensorflow(self, a: str, b: str, **kwargs: object) -> str:
        """Docstring."""
        return f"tf.linalg.matmul({a}, {b})"


@register_op("Dot")
class Dot(OpDef):
    """Docstring."""

    def infer_shape(self, a: object, b: object, **kwargs: object) -> object:
        """Docstring."""
        return None

    def numpy_eval(self, a: object, b: object, **kwargs: object) -> object:
        """Docstring."""
        return np.dot(a, b)

    def vjp(
        self, cotangent: object, a: object, b: object, **kwargs: object
    ) -> tuple[Any, ...]:
        """Docstring."""
        return (
            f"jnp.dot({cotangent}, {b})",
            f"jnp.dot({a}, {cotangent})",
        )  # Simplified

    def jvp(
        self,
        tangent_a: object,
        tangent_b: object,
        a: object,
        b: object,
        **kwargs: object,
    ) -> object:
        """Docstring."""
        return f"(jnp.dot({tangent_a}, {b}) + jnp.dot({a}, {tangent_b}))"

    def emit_jax(self, a: str, b: str, **kwargs: object) -> str:
        """Docstring."""
        return f"jnp.dot({a}, {b})"

    def emit_pytorch(self, a: str, b: str, **kwargs: object) -> str:
        """Docstring."""
        return f"torch.dot({a}, {b})"

    def emit_mlx(self, a: str, b: str, **kwargs: object) -> str:
        """Docstring."""
        return f"mx.dot({a}, {b})"  # Note MLX might use something else

    def emit_keras(self, a: str, b: str, **kwargs: object) -> str:
        """Docstring."""
        return f"keras.ops.dot({a}, {b})"

    def emit_tensorflow(self, a: str, b: str, **kwargs: object) -> str:
        """Docstring."""
        return f"tf.tensordot({a}, {b}, axes=1)"  # TF dot is tensordot with axes=1


@register_op("Einsum")
class Einsum(OpDef):
    """Docstring."""

    def infer_shape(
        self, subscripts: str, *operands: object, **kwargs: object
    ) -> object:
        """Docstring."""
        return None

    def numpy_eval(
        self, subscripts: str, *operands: object, **kwargs: object
    ) -> object:
        """Docstring."""
        return np.einsum(subscripts, *operands)

    def vjp(
        self, cotangent: object, subscripts: str, *operands: object, **kwargs: object
    ) -> tuple[Any, ...]:
        """Docstring."""
        # Einsum VJP is complex, returning placeholder
        return tuple([f"einsum_vjp({cotangent}, {i})" for i in range(len(operands))])

    def jvp(
        self, tangent: object, subscripts: str, *operands: object, **kwargs: object
    ) -> object:
        """Docstring."""
        return "einsum_jvp"

    def emit_jax(self, subscripts: str, *operands: str, **kwargs: object) -> str:
        """Docstring."""
        ops = ", ".join(operands)
        return f"jnp.einsum({subscripts}, {ops})"

    def emit_pytorch(self, subscripts: str, *operands: str, **kwargs: object) -> str:
        """Docstring."""
        ops = ", ".join(operands)
        return f"torch.einsum({subscripts}, {ops})"

    def emit_mlx(self, subscripts: str, *operands: str, **kwargs: object) -> str:
        """Docstring."""
        # MLX lacks full einsum, would need fallback, but we emit anyway
        ops = ", ".join(operands)
        return f"mx.einsum({subscripts}, {ops})"

    def emit_keras(self, subscripts: str, *operands: str, **kwargs: object) -> str:
        """Docstring."""
        ops = ", ".join(operands)
        return f"keras.ops.einsum({subscripts}, {ops})"

    def emit_tensorflow(self, subscripts: str, *operands: str, **kwargs: object) -> str:
        """Docstring."""
        ops = ", ".join(operands)
        return f"tf.einsum({subscripts}, {ops})"
