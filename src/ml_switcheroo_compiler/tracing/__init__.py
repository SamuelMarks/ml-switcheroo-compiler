"""Tracing module."""

from ml_switcheroo_compiler.tracing.tracer import (
    ProxyTensor,
    TracerTape,
    get_trace_count,
    global_tracing_state,
    increment_trace_count,
    reset_trace_count,
)

__all__ = [
    "ProxyTensor",
    "TracerTape",
    "get_trace_count",
    "global_tracing_state",
    "increment_trace_count",
    "reset_trace_count",
]
