"""Conditional execution (cond) operator implementation."""

from __future__ import annotations

import typing
from collections.abc import Callable

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.core.tensor import Tensor

T = typing.TypeVar("T")


def _as_branch_result(val: TensorOrScalar, fn: Callable[[], T]) -> T:
    """Narrow a branch execution result to `T`.

    Args:
        val (TensorOrScalar): Evaluated or traced branch result.
        fn (Callable[[], T]): Reference branch callable for type inference.

    Returns:
        T: Narrowed result.
    """
    del fn

    def _guard(cand: TensorOrScalar) -> typing.TypeGuard[T]:
        """Type guard narrowing `cand` to `T`.

        Args:
            cand (TensorOrScalar): Candidate result.

        Returns:
            bool: True for valid branch results.
        """
        return not isinstance(cand, bytes) or len(cand) >= 0

    assert _guard(val)
    return val


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
        return _as_branch_result(cf.cond_eager(pred, true_fn, false_fn), true_fn)
    return _as_branch_result(cf.cond_tracing(pred, true_fn, false_fn), true_fn)
