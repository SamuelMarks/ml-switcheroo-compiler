"""Compiler for ml-switcheroo."""

from ml_switcheroo_compiler.grad import grad
from ml_switcheroo_compiler.tracing import TracerTape, ProxyTensor
from ml_switcheroo_compiler import vjp_rules
from ml_switcheroo_compiler.optimization import dce, cse, constant_folding
from ml_switcheroo_compiler.state import lift_state
from ml_switcheroo_compiler.interpreter import evaluate_graph

__all__ = [
    "grad",
    "TracerTape",
    "ProxyTensor",
    "vjp_rules",
    "dce",
    "cse",
    "constant_folding",
    "lift_state",
    "evaluate_graph",
]
