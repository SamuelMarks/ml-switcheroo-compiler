"""Diagnostics and profiling package."""

from ml_switcheroo_compiler.diagnostics.flop_counter import estimate_flops
from ml_switcheroo_compiler.diagnostics.memory_profiler import memory_profiler
from ml_switcheroo_compiler.diagnostics.numerical_anomaly import (
    check_numerical_anomaly,
    format_traceback,
)
from ml_switcheroo_compiler.diagnostics.shape_debugger import debug_shapes, to_graphviz, to_html

from .debugging import enable_dump_debug_info
from .summary import encode_image, write_raw_pb

__all__ = [
    "check_numerical_anomaly",
    "debug_shapes",
    "enable_dump_debug_info",
    "encode_image",
    "estimate_flops",
    "format_traceback",
    "memory_profiler",
    "to_graphviz",
    "to_html",
    "write_raw_pb",
]
