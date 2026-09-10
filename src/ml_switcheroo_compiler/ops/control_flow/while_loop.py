"""While loop control flow operator implementation."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor

State = TypeVar("State")


def while_loop(
    cond_fn: Callable[[State], bool | Tensor],
    body_fn: Callable[[State], State],
    init_val: State,
) -> State:
    """Repeatedly apply body_fn to state while cond_fn evaluates to True.

    Dispatches to eager loop evaluation when eager mode is enabled, or records a
    Loop / WhileLoop control flow node into the active tracing IR graph.

    Args:
        cond_fn (Callable[[State], Union[bool, Tensor]]): Predicate callable determining whether to continue iterating.
        body_fn (Callable[[State], State]): Transition callable mapping current state to next state.
        init_val (State): Initial state value, tuple of values, or Tensor.

    Returns:
        State: Final accumulated state value after loop termination.
    """
    import ml_switcheroo_compiler.ops.control_flow as cf

    if config.eager_mode:
        return cf.while_loop_eager(cond_fn, body_fn, init_val)
    return cf.while_loop_tracing(cond_fn, body_fn, init_val)
