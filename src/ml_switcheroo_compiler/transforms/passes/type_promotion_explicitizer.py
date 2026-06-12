"""Type Promotion Explicitizer Pass."""

import uuid

import numpy as np
from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.ir.core import IRGraph
from ml_switcheroo_compiler.transforms.pass_manager import DAGTopologicalSorter


def type_promotion_explicitizer_pass(graph: IRGraph) -> bool:
    """In-place pass to explicitly inject Cast nodes."""
    modified = False
    sorted_nodes = DAGTopologicalSorter.sort(graph)

    from ml_switcheroo_compiler.transforms.passes.dtype_inference import dtype_inference_pass

    dtype_inference_pass(graph)

    for node in sorted_nodes:
        if len(node.inputs) != 2:
            continue

        in1, in2 = node.inputs
        dt1 = graph.nodes[in1].attributes.get("dtype")
        dt2 = graph.nodes[in2].attributes.get("dtype")

        if dt1 is None or dt2 is None or dt1 == dt2:
            continue

        try:
            target_dt = str(np.promote_types(np.dtype(dt1), np.dtype(dt2)))
        except TypeError:
            continue

        if dt1 != target_dt:
            new_id = f"cast_{uuid.uuid4().hex[:6]}"
            new_node = LogicalNode(
                id=new_id,
                op_type="Cast",
                inputs=[in1],
                shape_metadata=graph.nodes[in1].shape_metadata,
                attributes={"dtype": target_dt},
            )
            new_node.attributes["dtype"] = target_dt
            graph.nodes[new_id] = new_node
            node.inputs[0] = new_id
            modified = True

        if dt2 != target_dt:
            new_id = f"cast_{uuid.uuid4().hex[:6]}"
            new_node = LogicalNode(
                id=new_id,
                op_type="Cast",
                inputs=[in2],
                shape_metadata=graph.nodes[in2].shape_metadata,
                attributes={"dtype": target_dt},
            )
            new_node.attributes["dtype"] = target_dt
            graph.nodes[new_id] = new_node
            node.inputs[1] = new_id
            modified = True

    return modified
