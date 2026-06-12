"""Compiler for ml-switcheroo."""

from ml_switcheroo.core.tensor import Tensor
from ml_switcheroo.grad import grad
from ml_switcheroo.tracing import TracerTape, ProxyTensor
from ml_switcheroo import vjp_rules
from ml_switcheroo.state import lift_state
from ml_switcheroo.interpreter import evaluate_graph
from ml_switcheroo.core.config import EagerMode


__all__ = [
    "Tensor",
    "grad",
    "TracerTape",
    "ProxyTensor",
    "vjp_rules",
    "dce",
    "cse",
    "constant_folding",
    "lift_state",
    "evaluate_graph",
    "EagerMode",
]
