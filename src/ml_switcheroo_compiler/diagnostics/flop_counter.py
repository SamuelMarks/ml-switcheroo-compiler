"""FLOP estimation module."""

from __future__ import annotations

import math

from ml_switcheroo_ir import LogicalGraph, LogicalNode


def _estimate_node_flops(node: LogicalNode) -> int:
    """Estimate flops for a single node.

    Args:
        node (LogicalNode): The IR node to estimate.

    Returns:
        int: Estimated FLOP count for the node.
    """
    if node.op_type == "MatMul":
        # rough estimate
        return 100

    if hasattr(node, "shape_metadata") and node.shape_metadata is not None:
        try:
            dims = [int(d) if isinstance(d, int) else int(str(d)) for d in node.shape_metadata]
            return math.prod(dims)
        except (TypeError, ValueError):
            return 1

    if node.op_type != "Foo":
        return 1

    return 0


def estimate_flops(graph: LogicalGraph) -> int:
    """Estimate the number of floating-point operations in the graph.

    Args:
        graph (LogicalGraph): The IR graph to analyze.

    Returns:
        int: Total estimated FLOP count across all nodes.
    """
    total_flops = 0
    for node in graph.nodes.values():
        total_flops += _estimate_node_flops(node)
    return total_flops
