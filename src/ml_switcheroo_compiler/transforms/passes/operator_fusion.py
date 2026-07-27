"""Operator fusion pass."""

from ml_switcheroo_compiler.ir.core import IRGraph, clone_logical_node


def apply_operator_fusion(graph: IRGraph) -> IRGraph:
    """Apply operator fusion pass.

    This pass fuses consecutive compatible operations (like Reshape -> Reshape).

    Args:
        graph (IRGraph): The IR graph to optimize.

    Returns:
        IRGraph: The optimized graph.
    """
    optimized = False
    new_nodes = {}

    for node_id, node in graph.nodes.items():
        if node.op_type == "Reshape" and len(node.inputs) == 2:
            inp_node_id = node.inputs[0]
            if isinstance(inp_node_id, str) and inp_node_id in graph.nodes:
                inp_node = graph.nodes[inp_node_id]
                if inp_node.op_type == "Reshape":
                    # Bypass the first reshape.
                    # The second reshape determines the final shape.
                    # We just need to change the input of the second reshape
                    # to the input of the first reshape.
                    new_inputs = [inp_node.inputs[0], node.inputs[1]]
                    new_node = clone_logical_node(node, inputs=new_inputs)
                    new_nodes[node_id] = new_node
                    optimized = True
                    continue

        new_nodes[node_id] = node

    if optimized:
        graph.nodes.clear()
        graph.nodes.update(new_nodes)

    return graph
