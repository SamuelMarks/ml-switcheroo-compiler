"""Memory profiling module."""

import math
from ml_switcheroo_ir import LogicalGraph


def memory_profiler(graph: LogicalGraph) -> int:
    """Estimate memory usage of the graph.

    graph (LogicalGraph): The IR graph to analyze

    Returns:
    int: Estimated peak memory usage in bytes

    Args:
    graph (LogicalGraph): Argument graph
    """
    total_mem = 0
    for node in graph.nodes.values():
        if hasattr(node, "shape_metadata") and node.shape_metadata is not None:
            try:
                mem = int(math.prod(node.shape_metadata)) * 4  # Assume float32
                total_mem += mem
            except Exception:
                total_mem += 4
        else:
            total_mem += 4
    return total_mem
