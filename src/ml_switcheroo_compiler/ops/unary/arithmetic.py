# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Module arithmetic.py."""

"""Core abstractions and logic definitions for arithmetic.py."""

from ml_switcheroo_compiler.ops.base import register_op

from .base import UnaryMathOp


@register_op("Sqrt")
class Sqrt(UnaryMathOp):
    """Compute the non-negative square root element-wise."""

    op_name: object = "Sqrt"


@register_op("Square")
class Square(UnaryMathOp):
    """Compute the element-wise square of the input."""

    op_name: object = "Square"


@register_op("Abs")
class Abs(UnaryMathOp):
    """Compute the absolute value element-wise."""

    op_name: object = "Abs"


@register_op("Fabs")
class Fabs(UnaryMathOp):
    """Compute the absolute value element-wise, specifically returning floats."""

    op_name: object = "Fabs"
    np_op_name: object = "fabs"


@register_op("Negative")
class Negative(UnaryMathOp):
    """Compute the numerical negative element-wise."""

    op_name: object = "Negative"


@register_op("Positive")
class Positive(UnaryMathOp):
    """Compute the numerical positive element-wise (identity operation)."""

    op_name: object = "Positive"


@register_op("Sign")
class Sign(UnaryMathOp):
    """Compute an element-wise indication of the sign of a number (-1, 0, or 1)."""

    op_name: object = "Sign"


@register_op("Floor")
class Floor(UnaryMathOp):
    """Compute the floor of the input element-wise."""

    op_name: object = "Floor"


@register_op("Ceil")
class Ceil(UnaryMathOp):
    """Compute the ceiling of the input element-wise."""

    op_name: object = "Ceil"


@register_op("Round")
class Round(UnaryMathOp):
    """Round values of the input to the nearest integer element-wise."""

    op_name: object = "Round"


@register_op("Cbrt")
class Cbrt(UnaryMathOp):
    """Compute the cube root element-wise."""

    op_name: object = "Cbrt"
    np_op_name: object = "cbrt"


@register_op("Fix")
class Fix(UnaryMathOp):
    """Round elements to the nearest integer towards zero."""

    op_name: object = "Fix"
    np_op_name: object = "fix"


@register_op("Reciprocal")
class Reciprocal(UnaryMathOp):
    """Compute the reciprocal (1/x) element-wise."""

    op_name: object = "Reciprocal"
    np_op_name: object = "reciprocal"


@register_op("Trunc")
class Trunc(UnaryMathOp):
    """Truncates the fractional part of the input element-wise."""

    op_name: object = "Trunc"
    np_op_name: object = "trunc"


@register_op("Rsqrt")
class Rsqrt(UnaryMathOp):
    """Compute the reciprocal square root (1 / sqrt(x)) element-wise."""

    op_name: object = "Rsqrt"
    np_op_name: object = "rsqrt"


@register_op("Signbit")
class Signbit(UnaryMathOp):
    """Return True where signbit is set (less than zero)."""

    op_name: object = "Signbit"
    np_op_name: object = "signbit"


@register_op("ReciprocalNoNan")
class ReciprocalNoNan(UnaryMathOp):
    """ReciprocalNoNan operation."""

    op_name: object = "ReciprocalNoNan"


@register_op("Rint")
class Rint(UnaryMathOp):
    """Rint op."""

    op_name: object = "Rint"


@register_op("LogSigmoid")
class LogSigmoid(UnaryMathOp):
    """LogSigmoid operation."""

    op_name: object = "LogSigmoid"


@register_op("LogSoftmax")
class LogSoftmax(UnaryMathOp):
    """LogSoftmax operation."""

    op_name: object = "LogSoftmax"


@register_op("Softsign")
class Softsign(UnaryMathOp):
    """Softsign operation."""

    op_name: object = "Softsign"
