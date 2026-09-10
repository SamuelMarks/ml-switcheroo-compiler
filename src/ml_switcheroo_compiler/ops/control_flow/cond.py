"""Conditional execution (cond) operator implementation."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor

T = TypeVar("T")


def cond(
    pred: Tensor,
    true_fn: Callable[[], T],
    false_fn: Callable[[], T],
) -> T:
    """Conditionally execute either true_fn or false_fn based on predicate.

    Dispatches to eager execution when eager mode is enabled, or records an If/Cond
    control flow node into the active tracing IR graph.

    Args:
        pred (Tensor): A boolean scalar tensor determining the branch to execute.
        true_fn (Callable[[], T]): Nullary callable evaluated if pred is True.
        false_fn (Callable[[], T]): Nullary callable evaluated if pred is False.

    Returns:
        T: Result of executing the selected branch callable.
    """
    import ml_switcheroo_compiler.ops.control_flow as cf

    if config.eager_mode:
        return cf.cond_eager(pred, true_fn, false_fn)
    return cf.cond_tracing(pred, true_fn, false_fn)
