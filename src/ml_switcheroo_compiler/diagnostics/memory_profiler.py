# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Module memory_profiler.py."""

from typing import Any

"""Memory profiling module."""

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
                mem = int(math.prod(node.shape_metadata)) * 4  # Assume float32
                total_mem += mem
            except TypeError:
                total_mem += 4
        else:
            total_mem += 4
    return total_mem
