# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Provide mixin module."""

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.common.mixins.common import CommonASTVisitor

from .mixins.array import ArrayASTVisitor
from .mixins.control_flow import ControlFlowASTVisitor
from .mixins.distributed import DistributedASTVisitor
from .mixins.image import ImageASTVisitor
from .mixins.linalg import LinearAlgebraASTVisitor
from .mixins.variable import VariableASTVisitor


def get_shared_ast_visitors(generator: BaseGenerator) -> list[CommonASTVisitor]:
    """Return a list of shared AST visitors.

    Args:
        generator: The generator parameter.

    Returns:
        list[CommonASTVisitor]: A list of instantiated AST visitors.
    """
    return [
        ImageASTVisitor(generator=generator),
        ControlFlowASTVisitor(generator=generator),
        ArrayASTVisitor(generator=generator),
        VariableASTVisitor(generator=generator),
        LinearAlgebraASTVisitor(generator=generator),
        DistributedASTVisitor(generator=generator),
    ]
