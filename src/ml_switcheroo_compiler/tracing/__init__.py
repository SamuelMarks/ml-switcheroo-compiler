"""Tracing module."""

from ml_switcheroo_compiler.tracing.autograph import (
    LoopOptions,
    do_not_convert,
    set_loop_options,
)
from ml_switcheroo_compiler.tracing.tracer import (
    ProxyTensor,
    TracerTape,
    get_trace_count,
    global_tracing_state,
    increment_trace_count,
    reset_trace_count,
)

__all__ = [
    "LoopOptions",
    "do_not_convert",
    "set_loop_options",
    "ProxyTensor",
    "TracerTape",
    "get_trace_count",
    "global_tracing_state",
    "increment_trace_count",
    "reset_trace_count",
]
