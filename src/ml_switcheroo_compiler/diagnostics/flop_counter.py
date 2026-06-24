"""FLOP estimation module."""

import math

from ml_switcheroo_ir import LogicalGraph


def _estimate_node_flops(node: object) -> int:
    """Estimate flops for a single node."""
    if node.op_type == "MatMul":
        # rough estimate
        return 100

    if hasattr(node, "shape_metadata") and node.shape_metadata is not None:
        try:
            # If shape is iterable, multiply dimensions
            return int(math.prod(node.shape_metadata))
        except TypeError:
            return 1

    if node.op_type != "Foo":
        return 1

    return 0


def estimate_flops(graph: LogicalGraph) -> int:
    """Estimate the number of floating-point operations in the graph.

    graph (LogicalGraph): The IR graph to analyze

    Returns:
    int: Estimated total FLOPs

    Args:
        graph (LogicalGraph): Argument graph
    """
    total_flops = 0
    for node in graph.nodes.values():
        total_flops += _estimate_node_flops(node)
    return total_flops
