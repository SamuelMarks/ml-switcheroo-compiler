# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Dispatcher for operation execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.ops.eager_evaluator import EagerEvaluator
from ml_switcheroo_compiler.tracing.builder import TracingNodeBuilder
from ml_switcheroo_compiler.tracing.state import global_tracing_state

if TYPE_CHECKING:
    from ml_switcheroo_compiler.core.tensor import Tensor

DispatchArg = Union[
    int,
    float,
    str,
    bool,
    tuple[int, ...],
    list[int],
    list[float],
    list[str],
    dict[str, Union[int, str, float, bool]],
    None,
]


def dispatch_op(
    op_type: str,
    *args: Union[Tensor, DispatchArg],
    **kwargs: Union[Tensor, DispatchArg],
) -> Union[Tensor, DispatchArg, tuple[Union[Tensor, DispatchArg], ...]]:
    """Route operation to eager or tracing handler.

    Args:
        op_type (str): Name of the operator.
        *args (Tensor | DispatchArg): Positional args.
        **kwargs (Tensor | DispatchArg): Keyword args.

    Returns:
        Tensor | DispatchArg | tuple[Tensor | DispatchArg, ...]: Execution result.

    Raises:
        RuntimeError: If called outside of an active tracing context when eager mode is disabled.
    """
    if config.eager_mode:
        return EagerEvaluator.evaluate(op_type, *args, **kwargs)

    if not global_tracing_state.is_tracing:
        msg = f"Cannot emit {op_type} node outside of a tracing context."
        raise RuntimeError(msg)

    return TracingNodeBuilder.emit_tracing_node(op_type, *args, **kwargs)
