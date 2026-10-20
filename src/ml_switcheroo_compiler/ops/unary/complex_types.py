"""Implementation of complex types operations and utilities for the ops subsystem (`ml_switcheroo_compiler.ops.unary.complex_types`)."""

from __future__ import annotations

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
