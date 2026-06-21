"""Tracing module."""

from ml_switcheroo_compiler.tracing.tracer import (
    ProxyTensor,
    TracerTape,
    _tracer,
    get_trace_count,
    increment_trace_count,
    reset_trace_count,
)

__all__ = [
    "ProxyTensor",
    "TracerTape",
    "_tracer",
    "get_trace_count",
    "increment_trace_count",
    "reset_trace_count",
]
