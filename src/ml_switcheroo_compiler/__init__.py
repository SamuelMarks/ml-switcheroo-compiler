"""Compiler for ml-switcheroo."""

from ml_switcheroo_compiler.core.config import EagerMode
from ml_switcheroo_compiler.core.state_manager import lift_state
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.interpreter import evaluate_graph
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, TracerTape
from ml_switcheroo_compiler.transforms.autodiff import grad

__all__ = [
    "EagerMode",
    "ProxyTensor",
    "Tensor",
    "TracerTape",
    "evaluate_graph",
    "grad",
    "lift_state",
]
from .tree_util import tree_flatten, tree_map, tree_unflatten

__all__ += ["tree_flatten", "tree_map", "tree_unflatten"]
