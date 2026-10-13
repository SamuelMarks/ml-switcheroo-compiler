# ruff: noqa: E402, E501, F403
"""Module normalization.py."""

"""Core abstractions and logic definitions for normalization.py."""

from ml_switcheroo_compiler.ops.base import register_op

from .base import UnaryMathOp


@register_op("L2Normalize")
class L2Normalize(UnaryMathOp):
    """L2Normalize operation."""

    op_name = "L2Normalize"
