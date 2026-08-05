"""Type Promotion Explicitizer Pass."""

import uuid
from typing import Optional

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.core.constants import MAGIC_VAL_2
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.type_promotion import promote_types
from ml_switcheroo_compiler.ir.core import IRGraph
from ml_switcheroo_compiler.transforms.pass_manager import DAGTopologicalSorter
from ml_switcheroo_compiler.transforms.passes.dtype_inference import dtype_inference_pass


def _inject_cast_node(graph: IRGraph, input_id: str, target_dt: str) -> str:
    """Evaluate _inject_cast_node operation.

    Args:
        graph (IRGraph): The graph parameter.
        input_id (str): The input_id parameter.
        target_dt (str): The target_dt parameter.

    Returns:
        str: Result.
    """
    new_id = f"cast_{uuid.uuid4().hex[:6]}"
    new_node = LogicalNode(
        id=new_id,
        op_type="Cast",
        inputs=[input_id],
        shape_metadata=graph.nodes[input_id].shape_metadata,
        attributes={"dtype": target_dt},
    )
    graph.nodes[new_id] = new_node
    return new_id


def _needs_cast(dt1: Optional[str], dt2: Optional[str]) -> Optional[str]:
    """Evaluate _needs_cast operation.

    Args:
        dt1 (object): The dt1 parameter.
        dt2 (object): The dt2 parameter.

    Returns:
        object: Result.
    """
    if dt1 is None or dt2 is None or dt1 == dt2:
        return None
    try:
        return promote_types(DType(dt1), DType(dt2)).value
    except (TypeError, ValueError):
        return None


def type_promotion_explicitizer_pass(graph: IRGraph) -> bool:
    """In-place pass to explicitly inject Cast nodes.

    Args:
        graph (IRGraph): The graph parameter for the operation.

    Returns:
        bool: A boolean indicating the result of the check.
    """
    modified = False
    sorted_nodes = DAGTopologicalSorter.sort(graph)

    dtype_inference_pass(graph)

    for node in sorted_nodes:
        if len(node.inputs) != MAGIC_VAL_2:
            continue

        in1, in2 = node.inputs
        dt1 = graph.nodes[in1].attributes.get("dtype")
        dt2 = graph.nodes[in2].attributes.get("dtype")

        target_dt = _needs_cast(dt1, dt2)
        if target_dt is None:
            continue

        if dt1 != target_dt:
            node.inputs[0] = _inject_cast_node(graph, in1, target_dt)
            modified = True

        if dt2 != target_dt:
            node.inputs[1] = _inject_cast_node(graph, in2, target_dt)
            modified = True

    return modified
