"""Mixin module."""

from .mixins.array import ArrayASTVisitor
from .mixins.control_flow import ControlFlowASTVisitor
from .mixins.distributed import DistributedASTVisitor
from .mixins.image import ImageASTVisitor
from .mixins.linalg import LinearAlgebraASTVisitor
from .mixins.math import MathASTVisitor
from .mixins.nn import NNASTVisitor
from .mixins.variable import VariableASTVisitor


class SharedASTGeneratorVisitor(  # pylint: disable=too-many-ancestors
    MathASTVisitor,
    NNASTVisitor,
    ImageASTVisitor,
    ControlFlowASTVisitor,
    DistributedASTVisitor,
    ArrayASTVisitor,
    VariableASTVisitor,
    LinearAlgebraASTVisitor,
):
    """Shared AST generator mixin."""

    pass
