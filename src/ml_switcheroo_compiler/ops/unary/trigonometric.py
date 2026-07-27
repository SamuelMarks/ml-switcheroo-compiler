"""Core abstractions and logic definitions for trigonometric.py."""

from ml_switcheroo_compiler.ops.base import register_op

from .base import UnaryMathOp


@register_op("Sin")
class Sin(UnaryMathOp):
    """Computes the trigonometric sine element-wise."""

    op_name = "Sin"


@register_op("Cos")
class Cos(UnaryMathOp):
    """Computes the trigonometric cosine element-wise."""

    op_name = "Cos"


@register_op("Acos")
class Acos(UnaryMathOp):
    """Computes the inverse cosine (arccosine) element-wise."""

    op_name = "Acos"
    np_op_name = "arccos"


@register_op("Asin")
class Asin(UnaryMathOp):
    """Computes the inverse sine (arcsine) element-wise."""

    op_name = "Asin"
    np_op_name = "arcsin"


@register_op("Atan")
class Atan(UnaryMathOp):
    """Computes the inverse tangent (arctangent) element-wise."""

    op_name = "Atan"
    np_op_name = "arctan"


@register_op("Deg2Rad")
class Deg2Rad(UnaryMathOp):
    """Converts angles from degrees to radians element-wise."""

    op_name = "Deg2Rad"
    np_op_name = "deg2rad"


@register_op("Rad2Deg")
class Rad2Deg(UnaryMathOp):
    """Converts angles from radians to degrees element-wise."""

    op_name = "Rad2Deg"
    np_op_name = "rad2deg"


@register_op("Degrees")
class Degrees(UnaryMathOp):
    """Converts angles from radians to degrees element-wise."""

    op_name = "Degrees"
    np_op_name = "degrees"


@register_op("Radians")
class Radians(UnaryMathOp):
    """Converts angles from degrees to radians element-wise."""

    op_name = "Radians"
    np_op_name = "radians"


@register_op("Sinc")
class Sinc(UnaryMathOp):
    """Computes the normalized sinc function element-wise."""

    op_name = "Sinc"
    np_op_name = "sinc"


@register_op("Tan")
class Tan(UnaryMathOp):
    """Computes the trigonometric tangent element-wise."""

    op_name = "Tan"
    np_op_name = "tan"


@register_op("Angle")
class Angle(UnaryMathOp):
    """Computes the angle of the complex argument."""

    op_name = "Angle"
