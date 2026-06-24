"""Dispatcher for operation execution."""

from typing import Any
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.ops.eager_evaluator import EagerEvaluator
from ml_switcheroo_compiler.tracing.builder import TracingNodeBuilder
from ml_switcheroo_compiler.tracing import _tracer


def dispatch_op(op_type: str, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
    """Route operation to eager or tracing handler."""
    if config.eager_mode:
        return EagerEvaluator.evaluate(op_type, *args, **kwargs)

    if not _tracer.is_tracing:
        msg = f"Cannot emit {op_type} node outside of a tracing context."
        raise RuntimeError(msg)

    return TracingNodeBuilder.emit_tracing_node(op_type, *args, **kwargs)
