"""Compiler for ml-switcheroo."""

from ml_switcheroo_compiler.core.config import EagerMode
from ml_switcheroo_compiler.core.state_manager import lift_state
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.interpreter import evaluate_graph
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, TracerTape
from ml_switcheroo_compiler.transforms.autodiff import grad

from .tree_util import tree_flatten, tree_map, tree_unflatten

from ml_switcheroo_compiler.grad import (
    UnconnectedGradients,
    RegisterGradient,
    recompute_grad,
    custom_vjp,
    custom_jvp,
    jacobian,
    batch_jacobian,
    jacfwd,
    jacrev,
    hessian,
    jvp,
    vjp,
    linearize,
    linear_transpose,
)

__all__ = [
    "EagerMode",
    "ProxyTensor",
    "RegisterGradient",
    "Tensor",
    "TracerTape",
    "UnconnectedGradients",
    "batch_jacobian",
    "custom_jvp",
    "custom_vjp",
    "evaluate_graph",
    "grad",
    "hessian",
    "jacfwd",
    "jacobian",
    "jacrev",
    "jvp",
    "lift_state",
    "linear_transpose",
    "linearize",
    "recompute_grad",
    "tree_flatten",
    "tree_map",
    "tree_unflatten",
    "vjp",
]
