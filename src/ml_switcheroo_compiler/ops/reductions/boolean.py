from __future__ import annotations

# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
from typing import Any

"""Reductions."""


from ml_switcheroo_compiler.ops.base import register_op
from ml_switcheroo_compiler.ops.reductions.core import ReductionOp


@register_op("All")
class All(ReductionOp):
    """Apply logical AND reduction operation.

    Checks if all elements evaluate to True across specified dimensions of an input
    tensor
    """

    op_name = "All"
    np_op_name = "all"


@register_op("Any")
class AnyOp(ReductionOp):
    """Apply logical OR reduction operation.

    Checks if any elements evaluate to True across specified dimensions of an input
    tensor
    """

    op_name = "Any"
    np_op_name = "any"
