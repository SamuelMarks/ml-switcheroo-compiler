# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
from typing import Any

"""Core abstractions and logic definitions for normalization.py."""

from ml_switcheroo_compiler.ops.base import register_op

from .base import UnaryMathOp


@register_op("L2Normalize")
class L2Normalize(UnaryMathOp):
    """L2Normalize operation."""

    op_name = "L2Normalize"
