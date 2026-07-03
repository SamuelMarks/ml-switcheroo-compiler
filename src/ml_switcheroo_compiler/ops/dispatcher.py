"""Dispatcher for operation execution."""

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.ops.eager_evaluator import EagerEvaluator
from ml_switcheroo_compiler.tracing.builder import TracingNodeBuilder
from ml_switcheroo_compiler.tracing.state import global_tracing_state


def dispatch_op(op_type: str, *args: object, **kwargs: object) -> object:
    """Route operation to eager or tracing handler."""
    if config.eager_mode:
        return EagerEvaluator.evaluate(op_type, *args, **kwargs)

    if not global_tracing_state.is_tracing:
        msg = f"Cannot emit {op_type} node outside of a tracing context."
        raise RuntimeError(msg)

    return TracingNodeBuilder.emit_tracing_node(op_type, *args, **kwargs)
