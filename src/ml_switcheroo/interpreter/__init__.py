"""Interpreter package."""

from ml_switcheroo.interpreter.evaluator import evaluate_graph
from ml_switcheroo.interpreter.environment import Environment

__all__ = ["evaluate_graph", "Environment"]
