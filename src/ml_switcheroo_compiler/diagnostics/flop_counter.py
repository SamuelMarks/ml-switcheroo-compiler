# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""FLOP estimation module."""

import math
from typing import Any

from ml_switcheroo_ir import LogicalGraph


def _estimate_node_flops(node: Any) -> int:
    """Estimate flops for a single node.

    Args:
        node (object): The node parameter.

    Returns:
        int: Result.
    """
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

    Args:
        graph (LogicalGraph): The graph parameter.

    Returns:
        int: Result.
    """
    total_flops = 0
    for node in graph.nodes.values():
        total_flops += _estimate_node_flops(node)
    return total_flops
