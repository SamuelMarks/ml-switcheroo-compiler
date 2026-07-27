"""Core abstractions and logic definitions for complex_types.py."""

from ml_switcheroo_compiler.ops.base import register_op

from .base import UnaryMathOp


@register_op("Conj")
class Conj(UnaryMathOp):
    """Computes the complex conjugate element-wise."""

    op_name = "Conj"
    np_op_name = "conj"


@register_op("Imag")
class Imag(UnaryMathOp):
    """Returns the imaginary part of the complex argument element-wise."""

    op_name = "Imag"
    np_op_name = "imag"


@register_op("Real")
class Real(UnaryMathOp):
    """Returns the real part of the complex argument element-wise."""

    op_name = "Real"
    np_op_name = "real"
