"""State Lifting Pass."""

from ml_switcheroo_compiler.ir.core import IRGraph


def state_lifting_pass(graph: IRGraph) -> bool:
    """In-place pass to lift state to pure functional inputs.

    Args:
        graph (IRGraph): The graph.

    Returns:
        bool: The computed result.
    """
    modified = False

    # We find all ReadVariable and AssignVariable nodes
    # ReadVariable becomes an Input node
    # AssignVariable becomes an Output node (representing the mutated state)

    var_inputs = {}

    for nid, node in list(graph.nodes.items()):
        if node.op_type == "ReadVariable":
            var_name = node.attributes.get("variable_name", f"var_{nid}")
            node.op_type = "Input"
            node.attributes["name"] = var_name
            var_inputs[var_name] = nid
            modified = True

        elif node.op_type == "AssignVariable":
            var_name = node.attributes.get("variable_name", f"var_{nid}")
            node.inputs[0]

            # Change this node into an Output node for the new state
            node.op_type = "Output"
            node.attributes["name"] = f"{var_name}_out"
            # It just passes the value through

            # Make sure it's in the graph outputs
            if nid not in graph.outputs:
                graph.outputs.append(nid)

            modified = True

    return modified
