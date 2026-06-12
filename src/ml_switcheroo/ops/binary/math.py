"""Binary mathematical operations."""

import numpy as np
from typing import Any
from ml_switcheroo.ops.base import OpDef, register_op


class BinaryMathOp(OpDef):
    """Base class for binary mathematical operations."""

    op_name: str = ""

    def infer_shape(self, x: object, y: object, **kwargs: object) -> object:
        """Docstring."""
        # Broadcasting logic should ideally happen here, but for now we return x
        # This will be replaced by a proper shape inference pass.
        return (
            np.broadcast_shapes(x, y)
            if isinstance(x, tuple) and isinstance(y, tuple)
            else x
        )

    def numpy_eval(self, x: object, y: object, **kwargs: object) -> object:
        """Docstring."""
        return getattr(np, self.op_name.lower())(x, y)

    def emit_jax(self, x: str, y: str, **kwargs: object) -> str:
        """Docstring."""
        return f"jnp.{self.op_name.lower()}({x}, {y})"

    def emit_pytorch(self, x: str, y: str, **kwargs: object) -> str:
        """Docstring."""
        return f"torch.{self.op_name.lower()}({x}, {y})"

    def emit_mlx(self, x: str, y: str, **kwargs: object) -> str:
        """Docstring."""
        return f"mx.{self.op_name.lower()}({x}, {y})"

    def emit_keras(self, x: str, y: str, **kwargs: object) -> str:
        """Docstring."""
        return f"keras.ops.{self.op_name.lower()}({x}, {y})"

    def emit_tensorflow(self, x: str, y: str, **kwargs: object) -> str:
        """Docstring."""
        return f"tf.math.{self.op_name.lower()}({x}, {y})"


@register_op("Add")
class Add(BinaryMathOp):
    """Docstring."""

    op_name = "Add"

    def vjp(
        self, cotangent: object, x: object, y: object, **kwargs: object
    ) -> tuple[Any, ...]:
        """Docstring."""
        return (f"{cotangent}", f"{cotangent}")

    def jvp(
        self,
        tangent_x: object,
        tangent_y: object,
        x: object,
        y: object,
        **kwargs: object,
    ) -> object:
        """Docstring."""
        return f"({tangent_x} + {tangent_y})"


@register_op("Subtract")
class Subtract(BinaryMathOp):
    """Docstring."""

    op_name = "Subtract"

    def vjp(
        self, cotangent: object, x: object, y: object, **kwargs: object
    ) -> tuple[Any, ...]:
        """Docstring."""
        return (f"{cotangent}", f"-{cotangent}")

    def jvp(
        self,
        tangent_x: object,
        tangent_y: object,
        x: object,
        y: object,
        **kwargs: object,
    ) -> object:
        """Docstring."""
        return f"({tangent_x} - {tangent_y})"


@register_op("Multiply")
class Multiply(BinaryMathOp):
    """Docstring."""

    op_name = "Multiply"

    def vjp(
        self, cotangent: object, x: object, y: object, **kwargs: object
    ) -> tuple[Any, ...]:
        """Docstring."""
        return (f"({cotangent} * {y})", f"({cotangent} * {x})")

    def jvp(
        self,
        tangent_x: object,
        tangent_y: object,
        x: object,
        y: object,
        **kwargs: object,
    ) -> object:
        """Docstring."""
        return f"({tangent_x} * {y} + {x} * {tangent_y})"


@register_op("Divide")
class Divide(BinaryMathOp):
    """Docstring."""

    op_name = "Divide"

    def vjp(
        self, cotangent: object, x: object, y: object, **kwargs: object
    ) -> tuple[Any, ...]:
        """Docstring."""
        return (f"({cotangent} / {y})", f"(-{cotangent} * {x} / ({y} ** 2))")

    def jvp(
        self,
        tangent_x: object,
        tangent_y: object,
        x: object,
        y: object,
        **kwargs: object,
    ) -> object:
        """Docstring."""
        return f"(({tangent_x} * {y} - {x} * {tangent_y}) / ({y} ** 2))"


@register_op("TrueDivide")
class TrueDivide(Divide):
    """Docstring."""

    op_name = "True_Divide"

    def numpy_eval(self, x: object, y: object, **kwargs: object) -> object:
        """Docstring."""
        return np.true_divide(x, y)

    def emit_jax(self, x: str, y: str, **kwargs: object) -> str:
        """Docstring."""
        return f"jnp.true_divide({x}, {y})"

    def emit_pytorch(self, x: str, y: str, **kwargs: object) -> str:
        """Docstring."""
        return f"torch.true_divide({x}, {y})"

    def emit_mlx(self, x: str, y: str, **kwargs: object) -> str:
        """Docstring."""
        return f"mx.divide({x}, {y})"

    def emit_keras(self, x: str, y: str, **kwargs: object) -> str:
        """Docstring."""
        return f"keras.ops.true_divide({x}, {y})"

    def emit_tensorflow(self, x: str, y: str, **kwargs: object) -> str:
        """Docstring."""
        return f"tf.math.truediv({x}, {y})"


@register_op("Power")
class Power(BinaryMathOp):
    """Docstring."""

    op_name = "Power"

    def vjp(
        self, cotangent: object, x: object, y: object, **kwargs: object
    ) -> tuple[Any, ...]:
        """Docstring."""
        return (
            f"({cotangent} * {y} * jnp.power({x}, {y} - 1))",
            f"({cotangent} * jnp.power({x}, {y}) * jnp.log({x}))",
        )

    def jvp(
        self,
        tangent_x: object,
        tangent_y: object,
        x: object,
        y: object,
        **kwargs: object,
    ) -> object:
        """Docstring."""
        return f"({tangent_x} * {y} * jnp.power({x}, {y} - 1) + "
        f"{tangent_y} * jnp.power({x}, {y}) * jnp.log({x}))"


@register_op("Maximum")
class Maximum(BinaryMathOp):
    """Docstring."""

    op_name = "Maximum"

    def vjp(
        self, cotangent: object, x: object, y: object, **kwargs: object
    ) -> tuple[Any, ...]:
        """Docstring."""
        return (
            f"jnp.where({x} >= {y}, {cotangent}, 0)",
            f"jnp.where({x} < {y}, {cotangent}, 0)",
        )

    def jvp(
        self,
        tangent_x: object,
        tangent_y: object,
        x: object,
        y: object,
        **kwargs: object,
    ) -> object:
        """Docstring."""
        return f"jnp.where({x} >= {y}, {tangent_x}, {tangent_y})"


@register_op("Minimum")
class Minimum(BinaryMathOp):
    """Docstring."""

    op_name = "Minimum"

    def vjp(
        self, cotangent: object, x: object, y: object, **kwargs: object
    ) -> tuple[Any, ...]:
        """Docstring."""
        return (
            f"jnp.where({x} <= {y}, {cotangent}, 0)",
            f"jnp.where({x} > {y}, {cotangent}, 0)",
        )

    def jvp(
        self,
        tangent_x: object,
        tangent_y: object,
        x: object,
        y: object,
        **kwargs: object,
    ) -> object:
        """Docstring."""
        return f"jnp.where({x} <= {y}, {tangent_x}, {tangent_y})"
