"""Broadcast Explicitizer Pass."""

import uuid
from typing import Optional

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.core.shape import broadcast_shapes
from ml_switcheroo_compiler.ir.core import IRGraph
from ml_switcheroo_compiler.ops import get_op
from ml_switcheroo_compiler.transforms.pass_manager import DAGTopologicalSorter


def _inject_broadcast_node(graph: IRGraph, input_id: str, target_shape: tuple[int, ...]) -> str:
    new_id = f"broadcast_{uuid.uuid4().hex[:6]}"
    new_node = LogicalNode(
        id=new_id,
        op_type="BroadcastTo",
        inputs=[input_id],
        shape_metadata=target_shape,
        attributes={"shape": target_shape},
    )
    graph.nodes[new_id] = new_node
    return new_id


def _needs_broadcast(
    shape1: Optional[tuple[int, ...]], shape2: Optional[tuple[int, ...]]
) -> Optional[tuple[int, ...]]:
    if shape1 is None or shape2 is None or shape1 == shape2:
        return None
    try:
        return broadcast_shapes(shape1, shape2)
    except ValueError:
        return None


def broadcast_explicitizer_pass(graph: IRGraph) -> bool:
    """In-place pass to explicitly inject BroadcastTo nodes.

    Args:
        graph (IRGraph): The graph parameter for the operation.

    Returns:
        bool: A boolean indicating the result of the check.
    """
    modified = False
    sorted_nodes = DAGTopologicalSorter.sort(graph)

    # Needs shapes up to date
    from ml_switcheroo_compiler.transforms.passes.shape_inference import shape_inference_pass

    shape_inference_pass(graph)

    for node in sorted_nodes:
        try:
            get_op(node.op_type)
        except KeyError:
            continue

        if len(node.inputs) != 2:
            continue

        in1, in2 = node.inputs
        shape1 = graph.nodes[in1].shape_metadata
        shape2 = graph.nodes[in2].shape_metadata

        target_shape = _needs_broadcast(shape1, shape2)
        if target_shape is None:
            continue

        if shape1 != target_shape:
            node.inputs[0] = _inject_broadcast_node(graph, in1, target_shape)
            modified = True

        if shape2 != target_shape:
            node.inputs[1] = _inject_broadcast_node(graph, in2, target_shape)
            modified = True

    return modified
