"""Interpreter package."""

from __future__ import annotations

from ml_switcheroo_compiler.interpreter.environment import Environment
from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph
from ml_switcheroo_compiler.interpreter.typeclass_evaluator import evaluate_graph_with_typeclass

__all__ = [
    "Environment",
    "evaluate_graph",
    "evaluate_graph_with_typeclass",
]
