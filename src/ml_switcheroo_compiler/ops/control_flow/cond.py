"""Conditional execution (cond) operator implementation."""

from __future__ import annotations

from collections.abc import Callable

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.shape import OpValue
from ml_switcheroo_compiler.core.tensor import Tensor


def cond(
    pred: Tensor | OpValue,
    true_fn: Callable[[], OpValue] | OpValue,
    false_fn: Callable[[], OpValue] | OpValue,
) -> OpValue:
    """Conditionally execute either ``true_fn`` or ``false_fn`` based on predicate ``pred``.

    Dispatches to eager execution when eager mode is enabled, or records an ``If``
    control flow node into the active tracing IR graph.

    Args:
        pred (Tensor | OpValue): Boolean scalar tensor or predicate determining the branch to execute.
        true_fn (Callable[[], OpValue] | OpValue): Nullary callable evaluated if ``pred`` is True.
        false_fn (Callable[[], OpValue] | OpValue): Nullary callable evaluated if ``pred`` is False.

    Returns:
        OpValue: Result of executing the selected branch callable or traced output ``Tensor``.
    """
    import ml_switcheroo_compiler.ops.control_flow as cf
    from ml_switcheroo_compiler.core.tensor import ensure_tensor

    pred_t = pred if isinstance(pred, Tensor) else ensure_tensor(pred)
    if config.eager_mode:
        return cf.cond_eager(pred_t, true_fn, false_fn)
    return cf.cond_tracing(pred_t, true_fn, false_fn)
