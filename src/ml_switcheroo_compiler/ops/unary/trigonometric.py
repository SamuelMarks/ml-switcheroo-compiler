# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
from typing import Any

"""Core abstractions and logic definitions for trigonometric.py."""

from ml_switcheroo_compiler.ops.base import register_op

from .base import UnaryMathOp


@register_op("Sin")
class Sin(UnaryMathOp):
    """Compute the trigonometric sine element-wise."""

    op_name = "Sin"


@register_op("Cos")
class Cos(UnaryMathOp):
    """Compute the trigonometric cosine element-wise."""

    op_name = "Cos"


@register_op("Acos")
class Acos(UnaryMathOp):
    """Compute the inverse cosine (arccosine) element-wise."""

    op_name = "Acos"
    np_op_name = "arccos"


@register_op("Asin")
class Asin(UnaryMathOp):
    """Compute the inverse sine (arcsine) element-wise."""

    op_name = "Asin"
    np_op_name = "arcsin"


@register_op("Atan")
class Atan(UnaryMathOp):
    """Compute the inverse tangent (arctangent) element-wise."""

    op_name = "Atan"
    np_op_name = "arctan"


@register_op("Deg2Rad")
class Deg2Rad(UnaryMathOp):
    """Convert angles from degrees to radians element-wise."""

    op_name = "Deg2Rad"
    np_op_name = "deg2rad"


@register_op("Rad2Deg")
class Rad2Deg(UnaryMathOp):
    """Convert angles from radians to degrees element-wise."""

    op_name = "Rad2Deg"
    np_op_name = "rad2deg"


@register_op("Degrees")
class Degrees(UnaryMathOp):
    """Convert angles from radians to degrees element-wise."""

    op_name = "Degrees"
    np_op_name = "degrees"


@register_op("Radians")
class Radians(UnaryMathOp):
    """Convert angles from degrees to radians element-wise."""

    op_name = "Radians"
    np_op_name = "radians"


@register_op("Sinc")
class Sinc(UnaryMathOp):
    """Compute the normalized sinc function element-wise."""

    op_name = "Sinc"
    np_op_name = "sinc"


@register_op("Tan")
class Tan(UnaryMathOp):
    """Compute the trigonometric tangent element-wise."""

    op_name = "Tan"
    np_op_name = "tan"


@register_op("Angle")
class Angle(UnaryMathOp):
    """Compute the angle of the complex argument."""

    op_name = "Angle"
