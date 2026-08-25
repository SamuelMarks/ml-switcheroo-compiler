# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Module __init__.py."""

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
