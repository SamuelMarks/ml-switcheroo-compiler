"""Dispatcher for operation execution."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.ops.eager_evaluator import EagerEvaluator
from ml_switcheroo_compiler.tracing.builder import TracingNodeBuilder
from ml_switcheroo_compiler.tracing.state import global_tracing_state

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.core.tensor import Tensor

DispatchArg = TensorOrScalar


def dispatch_op(
    op_type: str,
    *args: Tensor | DispatchArg,
    **kwargs: Tensor | DispatchArg,
) -> Tensor | DispatchArg | tuple[Tensor | DispatchArg, ...]:
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
