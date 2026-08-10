# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
from typing import Any

"""Core abstractions and logic definitions for complex_types.py."""

from ml_switcheroo_compiler.ops.base import register_op

from .base import UnaryMathOp


@register_op("Conj")
class Conj(UnaryMathOp):
    """Compute the complex conjugate element-wise."""

    op_name = "Conj"
    np_op_name = "conj"


@register_op("Imag")
class Imag(UnaryMathOp):
    """Return the imaginary part of the complex argument element-wise."""

    op_name = "Imag"
    np_op_name = "imag"


@register_op("Real")
class Real(UnaryMathOp):
    """Return the real part of the complex argument element-wise."""

    op_name = "Real"
    np_op_name = "real"
