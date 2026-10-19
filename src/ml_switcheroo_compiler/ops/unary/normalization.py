"""Implementation of normalization operations and utilities for the ops subsystem (`ml_switcheroo_compiler.ops.unary.normalization`)."""

from ml_switcheroo_compiler.ops.base import register_op

from .base import UnaryMathOp


@register_op("L2Normalize")
class L2Normalize(UnaryMathOp):
    """L2Normalize operation."""

    op_name = "L2Normalize"
