# ruff: noqa: E501
"""Mixin module."""

from .mixins.array import ArrayASTVisitor
from .mixins.control_flow import ControlFlowASTVisitor
from .mixins.distributed import DistributedASTVisitor
from .mixins.image import ImageASTVisitor
from .mixins.linalg import LinearAlgebraASTVisitor
from .mixins.variable import VariableASTVisitor


def get_shared_ast_visitors(generator: object) -> list[object]:
    """Returns a list of shared AST visitors."""
    return [
        ImageASTVisitor(generator=generator),
        ControlFlowASTVisitor(generator=generator),
        DistributedASTVisitor(generator=generator),
        ArrayASTVisitor(generator=generator),
        VariableASTVisitor(generator=generator),
        LinearAlgebraASTVisitor(generator=generator),
    ]
