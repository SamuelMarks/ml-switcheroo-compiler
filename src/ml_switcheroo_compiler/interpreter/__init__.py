"""Interpreter package."""

from __future__ import annotations

from ml_switcheroo_compiler.interpreter.environment import Environment
from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph

__all__ = [
    "Environment",
    "evaluate_graph",
]
