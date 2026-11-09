"""Implementation of flop counter operations and utilities for the diagnostics subsystem (`ml_switcheroo_compiler.diagnostics.flop_counter`)."""

from __future__ import annotations

import math

from ml_switcheroo_ir import LogicalGraph, LogicalNode

_ZERO_FLOP_OPS: frozenset[str] = frozenset({"Input", "Output", "Constant", "NoOp", "Identity", "Reshape", "Squeeze", "ExpandDims", "Transpose"})


def _parse_shape_dims(node: LogicalNode | None) -> list[int] | None:
    """Extract positive integer dimensions from `node.shape_metadata` if valid.

    Args:
        node (LogicalNode | None): Logical IR node.

    Returns:
        list[int] | None: List of integer dimensions if available and concrete, else None.
    """
    if node is None or node.shape_metadata is None:
        return None
    try:
        return [int(d) if isinstance(d, int) else int(str(d)) for d in node.shape_metadata]
    except (TypeError, ValueError):
        return None


def _estimate_matmul_flops(node: LogicalNode, graph: LogicalGraph | None) -> int:
    """Estimate FLOPs for a `MatMul` or `Dot` node (`2 * M * N * K` with batch dimensions).

    Args:
        node (LogicalNode): The matrix multiplication node.
        graph (LogicalGraph | None): Optional parent graph for looking up input shapes.

    Returns:
        int: Estimated FLOP count.
    """
    out_dims = _parse_shape_dims(node)
    if graph is not None and len(node.inputs) >= 2:
        lhs_dims = _parse_shape_dims(graph.nodes.get(node.inputs[0]))
        rhs_dims = _parse_shape_dims(graph.nodes.get(node.inputs[1]))
        if lhs_dims and rhs_dims and len(lhs_dims) >= 2 and len(rhs_dims) >= 2:
            k_dim = lhs_dims[-1]
            if out_dims:
                return 2 * math.prod(out_dims) * k_dim
            batch = math.prod(lhs_dims[:-2]) if len(lhs_dims) > 2 else 1
            return 2 * batch * lhs_dims[-2] * rhs_dims[-1] * k_dim
    if out_dims:
        k_fallback = out_dims[-1] if len(out_dims) >= 2 else 1
        return 2 * math.prod(out_dims) * k_fallback
    return 1


def _estimate_conv2d_flops(node: LogicalNode, graph: LogicalGraph | None) -> int:
    """Estimate FLOPs for a `Conv2D` node (`2 * prod(out_shape) * C_in * K_h * K_w`).

    Args:
        node (LogicalNode): The 2D convolution node.
        graph (LogicalGraph | None): Optional parent graph for looking up weight shapes.

    Returns:
        int: Estimated FLOP count.
    """
    out_dims = _parse_shape_dims(node)
    if out_dims and graph is not None and len(node.inputs) >= 2:
        w_dims = _parse_shape_dims(graph.nodes.get(node.inputs[1]))
        if w_dims and len(w_dims) == 4:
            # Filter shape is (O, I, Kh, Kw) or (Kh, Kw, I, O)
            kernel_ops = w_dims[1] * w_dims[2] * w_dims[3] if w_dims[0] == out_dims[-1] or (len(out_dims) == 4 and w_dims[0] == out_dims[1]) else w_dims[0] * w_dims[1] * w_dims[2]
            return 2 * math.prod(out_dims) * kernel_ops
    if out_dims:
        return 2 * math.prod(out_dims)
    return 1


def _estimate_node_flops(node: LogicalNode, graph: LogicalGraph | None = None) -> int:
    """Estimate FLOPs for a single node based on its operation type and shape metadata.

    Args:
        node (LogicalNode): The IR node to estimate.
        graph (LogicalGraph | None): Optional parent graph to inspect input shapes.

    Returns:
        int: Estimated FLOP count for the node.
    """
    if node.op_type in _ZERO_FLOP_OPS:
        return 0

    if node.op_type in ("MatMul", "Dot"):
        return _estimate_matmul_flops(node, graph)

    if node.op_type == "Conv2D":
        return _estimate_conv2d_flops(node, graph)

    if node.shape_metadata is not None:
        dims = _parse_shape_dims(node)
        if dims is not None:
            return math.prod(dims)
        return 1

    return 1


def estimate_flops(graph: LogicalGraph) -> int:
    """Estimate the number of floating-point operations in the graph.

    Args:
        graph (LogicalGraph): The IR graph to analyze.

    Returns:
        int: Total estimated FLOP count across all nodes.
    """
    total_flops = 0
    for node in graph.nodes.values():
        total_flops += _estimate_node_flops(node, graph)
    return total_flops
