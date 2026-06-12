"""Diagnostics and profiling package."""

from ml_switcheroo.diagnostics.flop_counter import estimate_flops
from ml_switcheroo.diagnostics.memory_profiler import memory_profiler
from ml_switcheroo.diagnostics.shape_debugger import debug_shapes, to_graphviz, to_html
from ml_switcheroo.diagnostics.numerical_anomaly import (
    TracebackReconstructor,
    NumericalAnomalyDetector,
)

__all__ = [
    "estimate_flops",
    "memory_profiler",
    "debug_shapes",
    "to_graphviz",
    "to_html",
    "TracebackReconstructor",
    "NumericalAnomalyDetector",
]
