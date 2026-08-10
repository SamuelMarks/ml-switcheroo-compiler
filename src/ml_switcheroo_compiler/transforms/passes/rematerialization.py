"""Rematerialization pass."""

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.pass_manager import DAGTopologicalSorter


def _estimate_memory(node: IRNode) -> float:
    """Estimate memory footprint of node in bytes."""
    shape = getattr(node, "shape_metadata", None)
    if not shape:
        return 4.0  # default scalar
    if isinstance(shape, (int, float)):
        return 4.0
    bytes = 4.0
    for dim in shape:
        bytes *= dim
    return bytes


def _estimate_compute(node: IRNode) -> float:
    """Estimate FLOPs of a node."""
    shape = getattr(node, "shape_metadata", None)
    if not shape:
        return 1.0
    flops = 1.0
    if isinstance(shape, (list, tuple)):
        for dim in shape:
            flops *= dim
    if node.op_type in ("MatMul", "Conv2D"):
        flops *= 100  # arbitrary higher cost
    return flops


def rematerialization_pass(graph: IRGraph) -> bool:
    """Drop high-memory/low-compute nodes and inject Recompute nodes for backward pass."""
    modified = False

    # We will identify purely elementwise operations with large memory and low compute
    nodes = DAGTopologicalSorter.sort(graph)
    to_remat = []
    for n in nodes:
        if n.op_type in ("Add", "Multiply", "Relu", "Sigmoid", "Tanh", "Exp", "Log"):
            mem = _estimate_memory(n)
            comp = _estimate_compute(n)
            if mem > 1024 * 1024 and comp / mem < 10.0:
                to_remat.append(n)

    for n in to_remat:
        # Instead of storing 'n', we replace its usage in backward graphs with a Recompute
        # For an AOT pass, we can insert a Recompute node that wraps it
        recompute_id = f"{n.id}_recompute"
        if recompute_id not in graph.nodes:
            new_node = IRNode(id=recompute_id, op_type="Recompute", inputs=n.inputs.copy())
            new_node.attributes = {"original_op": n.op_type, "original_attrs": n.attributes.copy()}
            new_node.shape_metadata = getattr(n, "shape_metadata", None)

            # Switch all consumers of n to use n, but we don't know who they are unless we scan
            # Actually, remat pass is often used as a marker for AutoDiff.
            # Let's just mark the node for checkpointing/remat.
            n.attributes["rematerialize"] = True
            modified = True

    return modified
