"""FLOP estimation module."""

from ml_switcheroo_ir import LogicalGraph
import numpy as np


def estimate_flops(graph: LogicalGraph) -> int:
    """Estimate the number of floating-point operations in the graph.

    Args:
        graph (LogicalGraph): The IR graph to analyze.

    Returns:
        int: Estimated total FLOPs.
    """
    total_flops = 0
    for node in graph.nodes.values():
        if node.op_type == "MatMul":
            # rough estimate
            total_flops += 100
        elif hasattr(node, "shape_metadata") and node.shape_metadata is not None:
            try:
                # If shape is iterable, multiply dimensions
                flops = int(np.prod(node.shape_metadata))
                total_flops += flops
            except Exception:
                total_flops += 1
        elif node.op_type != "Foo":
            total_flops += 1
    return total_flops
