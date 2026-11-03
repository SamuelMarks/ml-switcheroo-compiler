"""Implementation of memory profiler operations and utilities for the diagnostics subsystem (`ml_switcheroo_compiler.diagnostics.memory_profiler`)."""

from __future__ import annotations

import math

from ml_switcheroo_ir import LogicalGraph

from ml_switcheroo_compiler.core.shape import _has_attr


def memory_profiler(graph: LogicalGraph) -> int:
    """Estimate memory usage of the graph.

    Args:
        graph (LogicalGraph): The IR graph to analyze.

    Returns:
        int: Estimated peak memory usage in bytes.
    """
    total_mem = 0
    for node in graph.nodes.values():
        if _has_attr(node, "shape_metadata") and node.shape_metadata is not None:
            try:
                dims = [int(d) if isinstance(d, int) else int(str(d)) for d in node.shape_metadata]
                mem = math.prod(dims) * 4  # Assume float32
                total_mem += mem
            except (TypeError, ValueError):
                total_mem += 4
        else:
            total_mem += 4
    return total_mem
