"""Reductions."""

from __future__ import annotations


from ml_switcheroo_compiler.ops.base import register_op


from ml_switcheroo_compiler.ops.reductions.core import ReductionOp


@register_op("All")
class All(ReductionOp):
    """Logical AND reduction operation.

    Checks if all elements evaluate to True across specified dimensions of an input
    tensor
    """

    op_name = "All"
    np_op_name = "all"


@register_op("Any")
class AnyOp(ReductionOp):
    """Logical OR reduction operation.

    Checks if any elements evaluate to True across specified dimensions of an input
    tensor
    """

    op_name = "Any"
    np_op_name = "any"
