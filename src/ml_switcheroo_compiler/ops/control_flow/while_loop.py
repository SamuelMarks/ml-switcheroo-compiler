"""While loop control flow operator implementation."""

from __future__ import annotations

import typing
from collections.abc import Callable

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.shape import OpValue
from ml_switcheroo_compiler.core.tensor import Tensor

State = typing.TypeVar("State")


@typing.overload
def while_loop(
    cond_fn: Callable[[Tensor], OpValue],
    body_fn: Callable[[Tensor], Tensor],
    init_val: Tensor,
) -> Tensor: ...


@typing.overload
def while_loop(
    cond_fn: Callable[[State], OpValue],
    body_fn: Callable[[State], State],
    init_val: State,
) -> State | Tensor: ...


def while_loop(
    cond_fn: Callable[[State], OpValue],
    body_fn: Callable[[State], State],
    init_val: State,
) -> State | Tensor:
    """Repeatedly apply ``body_fn`` to state while ``cond_fn`` evaluates to True.

    Dispatches to eager loop evaluation when eager mode is enabled, or records a
    ``Loop`` control flow node into the active tracing IR graph.

    Args:
        cond_fn (Callable[[State], OpValue]): Predicate callable determining whether to continue iterating.
        body_fn (Callable[[State], State]): Transition callable mapping current state to next state.
        init_val (State): Initial state value, tuple of values, or ``Tensor``.

    Returns:
        State | Tensor: Final accumulated state value or traced ``Tensor``.
    """
    import ml_switcheroo_compiler.ops.control_flow as cf

    if config.eager_mode:
        return cf.while_loop_eager(cond_fn, body_fn, init_val)
    return cf.while_loop_tracing(cond_fn, body_fn, init_val)
