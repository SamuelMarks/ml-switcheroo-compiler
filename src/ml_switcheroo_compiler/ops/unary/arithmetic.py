"""Core abstractions and logic definitions for arithmetic.py."""

from ml_switcheroo_compiler.ops.base import register_op

from .base import UnaryMathOp


@register_op("Sqrt")
class Sqrt(UnaryMathOp):
    """Compute the non-negative square root element-wise."""

    op_name = "Sqrt"


@register_op("Square")
class Square(UnaryMathOp):
    """Compute the element-wise square of the input."""

    op_name = "Square"


@register_op("Abs")
class Abs(UnaryMathOp):
    """Compute the absolute value element-wise."""

    op_name = "Abs"


@register_op("Fabs")
class Fabs(UnaryMathOp):
    """Compute the absolute value element-wise, specifically returning floats."""

    op_name = "Fabs"
    np_op_name = "fabs"


@register_op("Negative")
class Negative(UnaryMathOp):
    """Compute the numerical negative element-wise."""

    op_name = "Negative"


@register_op("Positive")
class Positive(UnaryMathOp):
    """Compute the numerical positive element-wise (identity operation)."""

    op_name = "Positive"


@register_op("Sign")
class Sign(UnaryMathOp):
    """Compute an element-wise indication of the sign of a number (-1, 0, or 1)."""

    op_name = "Sign"


@register_op("Floor")
class Floor(UnaryMathOp):
    """Compute the floor of the input element-wise."""

    op_name = "Floor"


@register_op("Ceil")
class Ceil(UnaryMathOp):
    """Compute the ceiling of the input element-wise."""

    op_name = "Ceil"


@register_op("Round")
class Round(UnaryMathOp):
    """Round values of the input to the nearest integer element-wise."""

    op_name = "Round"


@register_op("Cbrt")
class Cbrt(UnaryMathOp):
    """Compute the cube root element-wise."""

    op_name = "Cbrt"
    np_op_name = "cbrt"


@register_op("Fix")
class Fix(UnaryMathOp):
    """Round elements to the nearest integer towards zero."""

    op_name = "Fix"
    np_op_name = "fix"


@register_op("Reciprocal")
class Reciprocal(UnaryMathOp):
    """Compute the reciprocal (1/x) element-wise."""

    op_name = "Reciprocal"
    np_op_name = "reciprocal"


@register_op("Trunc")
class Trunc(UnaryMathOp):
    """Truncates the fractional part of the input element-wise."""

    op_name = "Trunc"
    np_op_name = "trunc"


@register_op("Rsqrt")
class Rsqrt(UnaryMathOp):
    """Compute the reciprocal square root (1 / sqrt(x)) element-wise."""

    op_name = "Rsqrt"
    np_op_name = "rsqrt"


@register_op("Signbit")
class Signbit(UnaryMathOp):
    """Return True where signbit is set (less than zero)."""

    op_name = "Signbit"
    np_op_name = "signbit"


@register_op("ReciprocalNoNan")
class ReciprocalNoNan(UnaryMathOp):
    """ReciprocalNoNan operation."""

    op_name = "ReciprocalNoNan"


@register_op("Rint")
class Rint(UnaryMathOp):
    """Rint op."""

    op_name = "Rint"


@register_op("LogSigmoid")
class LogSigmoid(UnaryMathOp):
    """LogSigmoid operation."""

    op_name = "LogSigmoid"


@register_op("LogSoftmax")
class LogSoftmax(UnaryMathOp):
    """LogSoftmax operation."""

    op_name = "LogSoftmax"


@register_op("Softsign")
class Softsign(UnaryMathOp):
    """Softsign operation."""

    op_name = "Softsign"
