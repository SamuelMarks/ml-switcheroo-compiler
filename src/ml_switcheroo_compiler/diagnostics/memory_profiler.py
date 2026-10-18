"""Memory profiling module."""

from __future__ import annotations

import math

from ml_switcheroo_ir import LogicalGraph


def memory_profiler(graph: LogicalGraph) -> int:
    """Estimate memory usage of the graph.

    Args:
        graph (LogicalGraph): The IR graph to analyze.

    Returns:
        int: Estimated peak memory usage in bytes.
    """
    total_mem = 0
    for node in graph.nodes.values():
        if hasattr(node, "shape_metadata") and node.shape_metadata is not None:
            try:
                dims = [int(d) if isinstance(d, int) else int(str(d)) for d in node.shape_metadata]
                mem = math.prod(dims) * 4  # Assume float32
                total_mem += mem
            except (TypeError, ValueError):
                total_mem += 4
        else:
            total_mem += 4
    return total_mem
