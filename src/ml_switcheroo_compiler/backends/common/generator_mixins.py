"""Provide mixin module."""

from typing import Callable, cast

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
        cast("Callable[..., CommonASTVisitor]", ImageASTVisitor)(generator=generator),
        cast("Callable[..., CommonASTVisitor]", ControlFlowASTVisitor)(generator=generator),
        cast("Callable[..., CommonASTVisitor]", ArrayASTVisitor)(generator=generator),
        cast("Callable[..., CommonASTVisitor]", VariableASTVisitor)(generator=generator),
        cast("Callable[..., CommonASTVisitor]", LinearAlgebraASTVisitor)(generator=generator),
        DistributedASTVisitor(generator=generator),
    ]
