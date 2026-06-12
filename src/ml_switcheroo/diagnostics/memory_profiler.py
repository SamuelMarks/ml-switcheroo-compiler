"""Memory profiling module."""

from ml_switcheroo_ir import LogicalGraph
import numpy as np


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
                mem = int(np.prod(node.shape_metadata)) * 4  # Assume float32
                total_mem += mem
            except Exception:
                total_mem += 4
        else:
            total_mem += 4
    return total_mem
