"""Broadcast Explicitizer Pass."""

import uuid

import numpy as np
from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.ir.core import IRGraph
from ml_switcheroo_compiler.ops import get_op
from ml_switcheroo_compiler.transforms.pass_manager import DAGTopologicalSorter


def broadcast_explicitizer_pass(graph: IRGraph) -> bool:
    """In-place pass to explicitly inject BroadcastTo nodes."""
    modified = False
    sorted_nodes = DAGTopologicalSorter.sort(graph)

    # Needs shapes up to date
    from ml_switcheroo_compiler.transforms.passes.shape_inference import shape_inference_pass

    shape_inference_pass(graph)

    for node in sorted_nodes:
        # Only binary ops typically broadcast implicitly
        try:
            get_op(node.op_type)
        except KeyError:
            continue

        # Is it a binary op? Check if it inherits from BinaryMathOp
        # Or just check if inputs == 2
        if len(node.inputs) != 2:
            continue

        in1, in2 = node.inputs
        shape1 = graph.nodes[in1].shape_metadata
        shape2 = graph.nodes[in2].shape_metadata

        # Don't try broadcasting None shapes
        if shape1 is None or shape2 is None:
            continue

        if shape1 == shape2:
            continue

        try:
            target_shape = np.broadcast_shapes(shape1, shape2)
        except ValueError:
            # Shapes are fundamentally incompatible. Shape pass or execution will fail
            continue

        if shape1 != target_shape:
            # Inject BroadcastTo for input 1
            new_id = f"broadcast_{uuid.uuid4().hex[:6]}"
            new_node = LogicalNode(
                id=new_id,
                op_type="BroadcastTo",
                inputs=[in1],
                shape_metadata=target_shape,
                attributes={"shape": target_shape},
            )
            graph.nodes[new_id] = new_node
            node.inputs[0] = new_id
            modified = True

        if shape2 != target_shape:
            # Inject BroadcastTo for input 2
            new_id = f"broadcast_{uuid.uuid4().hex[:6]}"
            new_node = LogicalNode(
                id=new_id,
                op_type="BroadcastTo",
                inputs=[in2],
                shape_metadata=target_shape,
                attributes={"shape": target_shape},
            )
            graph.nodes[new_id] = new_node
            node.inputs[1] = new_id
            modified = True

    return modified
