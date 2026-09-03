# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Dispatcher for operation execution."""

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.ops.eager_evaluator import EagerEvaluator
from ml_switcheroo_compiler.tracing.builder import TracingNodeBuilder
from ml_switcheroo_compiler.tracing.state import global_tracing_state


def dispatch_op(op_type: str, *args, **kwargs):
    """Route operation to eager or tracing handler.

    Args:
        op_type (str): The op_type parameter.
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            tuple[int, ...]: Result.

    Raises:
        RuntimeError: An exception.
    """
    if config.eager_mode:
        return EagerEvaluator.evaluate(op_type, *args, **kwargs)

    if not global_tracing_state.is_tracing:
        msg = f"Cannot emit {op_type} node outside of a tracing context."
        raise RuntimeError(msg)

    return TracingNodeBuilder.emit_tracing_node(op_type, *args, **kwargs)
