# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Module complex_types.py."""

"""Core abstractions and logic definitions for complex_types.py."""

from ml_switcheroo_compiler.ops.base import register_op

from .base import UnaryMathOp


@register_op("Conj")
class Conj(UnaryMathOp):
    """Compute the complex conjugate element-wise."""

    op_name: object = "Conj"
    np_op_name: object = "conj"


@register_op("Imag")
class Imag(UnaryMathOp):
    """Return the imaginary part of the complex argument element-wise."""

    op_name: object = "Imag"
    np_op_name: object = "imag"


@register_op("Real")
class Real(UnaryMathOp):
    """Return the real part of the complex argument element-wise."""

    op_name: object = "Real"
    np_op_name: object = "real"
