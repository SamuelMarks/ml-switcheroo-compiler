"""Unary mathematical operations."""

import numpy as np
from typing import Any
from ml_switcheroo.ops.base import OpDef, register_op


class UnaryMathOp(OpDef):
    """Base class for unary mathematical operations."""

    op_name: str = ""

    def infer_shape(self, x: object, **kwargs: object) -> object:
        """Docstring."""
        return x  # Unary ops typically preserve shape and dtype

    def numpy_eval(self, x: object, **kwargs: object) -> object:
        """Docstring."""
        return getattr(np, self.op_name.lower())(x)

    def emit_jax(self, x: str, **kwargs: object) -> str:
        """Docstring."""
        return f"jnp.{self.op_name.lower()}({x})"

    def emit_pytorch(self, x: str, **kwargs: object) -> str:
        """Docstring."""
        return f"torch.{self.op_name.lower()}({x})"

    def emit_mlx(self, x: str, **kwargs: object) -> str:
        """Docstring."""
        return f"mx.{self.op_name.lower()}({x})"

    def emit_keras(self, x: str, **kwargs: object) -> str:
        """Docstring."""
        return f"keras.ops.{self.op_name.lower()}({x})"

    def emit_tensorflow(self, x: str, **kwargs: object) -> str:
        """Docstring."""
        return f"tf.math.{self.op_name.lower()}({x})"


@register_op("Sin")
class Sin(UnaryMathOp):
    """Docstring."""

    op_name = "Sin"

    def vjp(self, cotangent: object, x: object, **kwargs: object) -> tuple[Any, ...]:
        """Docstring."""
        # d(sin(x)) = cos(x) * dx
        return (f"({cotangent} * jnp.cos({x}))",)

    def jvp(self, tangent: object, x: object, **kwargs: object) -> object:
        """Docstring."""
        return f"({tangent} * jnp.cos({x}))"


@register_op("Cos")
class Cos(UnaryMathOp):
    """Docstring."""

    op_name = "Cos"

    def vjp(self, cotangent: object, x: object, **kwargs: object) -> tuple[Any, ...]:
        """Docstring."""
        # d(cos(x)) = -sin(x) * dx
        return (f"({cotangent} * -jnp.sin({x}))",)

    def jvp(self, tangent: object, x: object, **kwargs: object) -> object:
        """Docstring."""
        return f"({tangent} * -jnp.sin({x}))"


@register_op("Exp")
class Exp(UnaryMathOp):
    """Docstring."""

    op_name = "Exp"

    def vjp(self, cotangent: object, x: object, **kwargs: object) -> tuple[Any, ...]:
        """Docstring."""
        return (f"({cotangent} * jnp.exp({x}))",)

    def jvp(self, tangent: object, x: object, **kwargs: object) -> object:
        """Docstring."""
        return f"({tangent} * jnp.exp({x}))"


@register_op("Log")
class Log(UnaryMathOp):
    """Docstring."""

    op_name = "Log"

    def vjp(self, cotangent: object, x: object, **kwargs: object) -> tuple[Any, ...]:
        """Docstring."""
        return (f"({cotangent} / {x})",)

    def jvp(self, tangent: object, x: object, **kwargs: object) -> object:
        """Docstring."""
        return f"({tangent} / {x})"


@register_op("Sqrt")
class Sqrt(UnaryMathOp):
    """Docstring."""

    op_name = "Sqrt"

    def vjp(self, cotangent: object, x: object, **kwargs: object) -> tuple[Any, ...]:
        """Docstring."""
        return (f"({cotangent} / (2 * jnp.sqrt({x})))",)

    def jvp(self, tangent: object, x: object, **kwargs: object) -> object:
        """Docstring."""
        return f"({tangent} / (2 * jnp.sqrt({x})))"


@register_op("Square")
class Square(UnaryMathOp):
    """Docstring."""

    op_name = "Square"

    def vjp(self, cotangent: object, x: object, **kwargs: object) -> tuple[Any, ...]:
        """Docstring."""
        return (f"({cotangent} * 2 * {x})",)

    def jvp(self, tangent: object, x: object, **kwargs: object) -> object:
        """Docstring."""
        return f"({tangent} * 2 * {x})"


@register_op("Abs")
class Abs(UnaryMathOp):
    """Docstring."""

    op_name = "Abs"

    def vjp(self, cotangent: object, x: object, **kwargs: object) -> tuple[Any, ...]:
        """Docstring."""
        return (f"({cotangent} * jnp.sign({x}))",)

    def jvp(self, tangent: object, x: object, **kwargs: object) -> object:
        """Docstring."""
        return f"({tangent} * jnp.sign({x}))"


@register_op("Negative")
class Negative(UnaryMathOp):
    """Docstring."""

    op_name = "Negative"

    def vjp(self, cotangent: object, x: object, **kwargs: object) -> tuple[Any, ...]:
        """Docstring."""
        return (f"-{cotangent}",)

    def jvp(self, tangent: object, x: object, **kwargs: object) -> object:
        """Docstring."""
        return f"-{tangent}"


@register_op("Positive")
class Positive(UnaryMathOp):
    """Docstring."""

    op_name = "Positive"

    def vjp(self, cotangent: object, x: object, **kwargs: object) -> tuple[Any, ...]:
        """Docstring."""
        return (f"{cotangent}",)

    def jvp(self, tangent: object, x: object, **kwargs: object) -> object:
        """Docstring."""
        return f"{tangent}"


@register_op("Sign")
class Sign(UnaryMathOp):
    """Docstring."""

    op_name = "Sign"

    def vjp(self, cotangent: object, x: object, **kwargs: object) -> tuple[Any, ...]:
        """Docstring."""
        return ("0",)

    def jvp(self, tangent: object, x: object, **kwargs: object) -> object:
        """Docstring."""
        return "0"


@register_op("Floor")
class Floor(UnaryMathOp):
    """Docstring."""

    op_name = "Floor"

    def vjp(self, cotangent: object, x: object, **kwargs: object) -> tuple[Any, ...]:
        """Docstring."""
        return ("0",)

    def jvp(self, tangent: object, x: object, **kwargs: object) -> object:
        """Docstring."""
        return "0"


@register_op("Ceil")
class Ceil(UnaryMathOp):
    """Docstring."""

    op_name = "Ceil"

    def vjp(self, cotangent: object, x: object, **kwargs: object) -> tuple[Any, ...]:
        """Docstring."""
        return ("0",)

    def jvp(self, tangent: object, x: object, **kwargs: object) -> object:
        """Docstring."""
        return "0"


@register_op("Round")
class Round(UnaryMathOp):
    """Docstring."""

    op_name = "Round"

    def vjp(self, cotangent: object, x: object, **kwargs: object) -> tuple[Any, ...]:
        """Docstring."""
        return ("0",)

    def jvp(self, tangent: object, x: object, **kwargs: object) -> object:
        """Docstring."""
        return "0"
