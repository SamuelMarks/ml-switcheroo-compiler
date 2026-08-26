# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Module batch_norm_folding.py."""

"""Batch Norm Folding pass."""

from ml_switcheroo_compiler.ir.core import IRGraph
from ml_switcheroo_compiler.transforms.pass_manager import DAGTopologicalSorter


def batch_norm_folding_pass(graph: IRGraph) -> bool:
    """In-place Batch Norm Folding pass.

    Folds BatchNorm operations into preceding Conv2D operations by rewiring the graph.

    Args:
        graph (IRGraph): The input graph to mutate.

    Returns:
        bool: True if the graph was modified, False otherwise.
    """
    modified = False

    import uuid

    from ml_switcheroo_compiler.ir.core import IRNode

    sorted_nodes = DAGTopologicalSorter.sort(graph)
    new_nodes = dict(graph.nodes)

    for node in sorted_nodes:
        if node.op_type == "BatchNorm":
            if node.inputs:
                prev_node = graph.nodes.get(node.inputs[0])
                if prev_node and prev_node.op_type == "Conv2D":
                    # Collect batchnorm inputs (moving mean, var, scale, bias)
                    # We fold them by inserting math nodes to update Conv2D's W and B
                    # W_new = W * gamma / sqrt(var + eps)
                    # B_new = (B - mean) * gamma / sqrt(var + eps) + beta

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

                    if b:
                        sub_id = f"bn_sub_{uid}"
                        new_nodes[sub_id] = IRNode(id=sub_id, op_type="Subtract", inputs=[b, mean])
                        b_mult_id = f"bn_b_mult_{uid}"
                        new_nodes[b_mult_id] = IRNode(id=b_mult_id, op_type="Multiply", inputs=[sub_id, multiplier_id])
                        b_new_id = f"bn_b_new_{uid}"
                        new_nodes[b_new_id] = IRNode(id=b_new_id, op_type="Add", inputs=[b_mult_id, beta])
                    else:
                        sub_id = f"bn_sub_{uid}"
                        zero_id = f"bn_zero_{uid}"
                        new_nodes[zero_id] = IRNode(id=zero_id, op_type="Constant", attributes={"value": 0.0})
                        new_nodes[sub_id] = IRNode(id=sub_id, op_type="Subtract", inputs=[zero_id, mean])
                        b_mult_id = f"bn_b_mult_{uid}"
                        new_nodes[b_mult_id] = IRNode(id=b_mult_id, op_type="Multiply", inputs=[sub_id, multiplier_id])
                        b_new_id = f"bn_b_new_{uid}"
                        new_nodes[b_new_id] = IRNode(id=b_new_id, op_type="Add", inputs=[b_mult_id, beta])

                    # Update Conv2D node inputs
                    prev_node.inputs = [prev_node.inputs[0], w_new_id, b_new_id] + prev_node.inputs[3:]

                    # Rewire consumers of BatchNorm to point to Conv2D
                    for other_node in new_nodes.values():
                        if node.id in other_node.inputs:
                            other_node.inputs = [prev_node.id if i == node.id else i for i in other_node.inputs]
                            modified = True

                    if node.id in graph.outputs:
                        graph.outputs = [prev_node.id if o == node.id else o for o in graph.outputs]
                        modified = True

                    # Remove the batch norm node from the graph
                    del new_nodes[node.id]
                    modified = True

    if modified:
        graph.nodes.clear()
        graph.nodes.update(new_nodes)

    return modified
