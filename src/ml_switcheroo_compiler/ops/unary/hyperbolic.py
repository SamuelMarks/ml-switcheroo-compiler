"""Core abstractions and logic definitions for hyperbolic.py."""

from ml_switcheroo_compiler.ops.base import register_op

from .base import UnaryMathOp


@register_op("Acosh")
class Acosh(UnaryMathOp):
    """Compute the inverse hyperbolic cosine element-wise."""

    op_name = "Acosh"
    np_op_name = "arccosh"


@register_op("Asinh")
class Asinh(UnaryMathOp):
    """Compute the inverse hyperbolic sine element-wise."""

    op_name = "Asinh"
    np_op_name = "arcsinh"


@register_op("Atanh")
class Atanh(UnaryMathOp):
    """Compute the inverse hyperbolic tangent element-wise."""

    op_name = "Atanh"
    np_op_name = "arctanh"


@register_op("Cosh")
class Cosh(UnaryMathOp):
    """Compute the hyperbolic cosine element-wise."""

    op_name = "Cosh"
    np_op_name = "cosh"


@register_op("Sinh")
class Sinh(UnaryMathOp):
    """Compute the hyperbolic sine element-wise."""

    op_name = "Sinh"
    np_op_name = "sinh"


@register_op("Tanh")
class Tanh(UnaryMathOp):
    """Compute the hyperbolic tangent element-wise."""

    op_name = "Tanh"
    np_op_name = "tanh"
