# ruff: noqa: E501
"""Provide mixin module."""

from .mixins.array import ArrayASTVisitor
from .mixins.control_flow import ControlFlowASTVisitor
from .mixins.distributed import DistributedASTVisitor
from .mixins.image import ImageASTVisitor
from .mixins.linalg import LinearAlgebraASTVisitor
from .mixins.variable import VariableASTVisitor


def get_shared_ast_visitors(generator: object) -> list[object]:
    """Return a list of shared AST visitors.

    Args:
        generator (object): The generator parameter.

    Returns:
        object: Result.
    """
    return [
        ImageASTVisitor(generator=generator),
        ControlFlowASTVisitor(generator=generator),
        DistributedASTVisitor(generator=generator),
        ArrayASTVisitor(generator=generator),
        VariableASTVisitor(generator=generator),
        LinearAlgebraASTVisitor(generator=generator),
    ]
