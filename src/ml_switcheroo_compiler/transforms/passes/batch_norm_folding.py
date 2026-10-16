"""Batch Norm Folding pass."""

from __future__ import annotations

import uuid

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.pass_manager import DAGTopologicalSorter


def _fold_bn_into_conv(
    node: IRNode,
    prev_node: IRNode,
    new_nodes: dict[str, IRNode],
    graph: IRGraph,
) -> None:
    """Fold a BatchNorm node into its preceding Conv2D node.

    Args:
        node (IRNode): The BatchNorm node to fold.
        prev_node (IRNode): The preceding Conv2D node.
        new_nodes (dict[str, IRNode]): Mutable mapping of graph nodes.
        graph (IRGraph): The computation graph being mutated.
    """
    gamma, beta, mean, var = node.inputs[1:5]
    eps_val = node.attributes.get("epsilon", 1e-5)
    w = prev_node.inputs[1]
    b = prev_node.inputs[2] if len(prev_node.inputs) > 2 else None

    uid = uuid.uuid4().hex[:6]

    eps_id = f"bn_eps_{uid}"
    new_nodes[eps_id] = IRNode(id=eps_id, op_type="Constant", attributes={"value": eps_val})

    var_eps_id = f"bn_var_eps_{uid}"
    new_nodes[var_eps_id] = IRNode(id=var_eps_id, op_type="Add", inputs=[var, eps_id])

    sqrt_id = f"bn_sqrt_{uid}"
    new_nodes[sqrt_id] = IRNode(id=sqrt_id, op_type="Sqrt", inputs=[var_eps_id])

    multiplier_id = f"bn_mult_{uid}"
    new_nodes[multiplier_id] = IRNode(id=multiplier_id, op_type="Divide", inputs=[gamma, sqrt_id])

    w_new_id = f"bn_w_new_{uid}"
    new_nodes[w_new_id] = IRNode(id=w_new_id, op_type="Multiply", inputs=[w, multiplier_id])

    bias_source_id = b
    if not bias_source_id:
        bias_source_id = f"bn_zero_{uid}"
        new_nodes[bias_source_id] = IRNode(id=bias_source_id, op_type="Constant", attributes={"value": 0.0})

    sub_id = f"bn_sub_{uid}"
    new_nodes[sub_id] = IRNode(id=sub_id, op_type="Subtract", inputs=[bias_source_id, mean])
    b_mult_id = f"bn_b_mult_{uid}"
    new_nodes[b_mult_id] = IRNode(id=b_mult_id, op_type="Multiply", inputs=[sub_id, multiplier_id])
    b_new_id = f"bn_b_new_{uid}"
    new_nodes[b_new_id] = IRNode(id=b_new_id, op_type="Add", inputs=[b_mult_id, beta])

    prev_node.inputs = [prev_node.inputs[0], w_new_id, b_new_id] + prev_node.inputs[3:]

    for other_node in new_nodes.values():
        if node.id in other_node.inputs:
            other_node.inputs = [prev_node.id if i == node.id else i for i in other_node.inputs]

    if node.id in graph.outputs:
        graph.outputs = [prev_node.id if o == node.id else o for o in graph.outputs]

    del new_nodes[node.id]


def batch_norm_folding_pass(graph: IRGraph) -> bool:
    """In-place Batch Norm Folding pass.

    Folds BatchNorm operations into preceding Conv2D operations by rewiring the graph.

    Args:
        graph (IRGraph): The input graph to mutate.

    Returns:
        bool: True if the graph was modified, False otherwise.
    """
    modified = False
    sorted_nodes = DAGTopologicalSorter.sort(graph)
    new_nodes = dict(graph.nodes)

    for node in sorted_nodes:
        if node.op_type == "BatchNorm" and node.inputs:
            prev_node = graph.nodes.get(node.inputs[0])
            if prev_node and prev_node.op_type == "Conv2D":
                _fold_bn_into_conv(node, prev_node, new_nodes, graph)
                modified = True

    if modified:
        graph.nodes.clear()
        graph.nodes.update(new_nodes)

    return modified
