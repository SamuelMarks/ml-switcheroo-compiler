"""Implementation of memory profiler operations and utilities for the diagnostics subsystem (`ml_switcheroo_compiler.diagnostics.memory_profiler`)."""

from __future__ import annotations

import math

from ml_switcheroo_ir import LogicalGraph, LogicalNode

_DTYPE_BYTE_SIZES: dict[str, int] = {
    "bool": 1,
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "uint16": 2,
    "float16": 2,
    "bfloat16": 2,
    "int32": 4,
    "uint32": 4,
    "float32": 4,
    "int64": 8,
    "uint64": 8,
    "float64": 8,
    "complex64": 8,
    "complex128": 16,
}


def _resolve_node_itemsize(node: LogicalNode) -> int:
    """Resolve the byte width per element from `node.dtype` or `node.attributes['dtype']`.

    Args:
        node (LogicalNode): Target IR node.

    Returns:
        int: Number of bytes per element (defaults to 4 for float32).
    """
    raw_dtype = node.dtype if node.dtype else node.attributes.get("dtype", "float32")
    dt_str = str(raw_dtype).lower()
    if dt_str.startswith("dtype."):
        dt_str = dt_str.split(".", 1)[1]
    return _DTYPE_BYTE_SIZES.get(dt_str, 4)


def memory_profiler(graph: LogicalGraph) -> int:
    """Estimate memory usage of the graph in bytes across all nodes.

    Args:
        graph (LogicalGraph): The IR graph to analyze.

    Returns:
        int: Estimated total memory usage in bytes.
    """
    total_mem = 0
    for node in graph.nodes.values():
        itemsize = _resolve_node_itemsize(node)
        if node.shape_metadata is not None:
            try:
                dims = [int(d) if isinstance(d, int) else int(str(d)) for d in node.shape_metadata]
                total_mem += math.prod(dims) * itemsize
            except (TypeError, ValueError):
                total_mem += itemsize
        else:
            total_mem += itemsize
    return total_mem
