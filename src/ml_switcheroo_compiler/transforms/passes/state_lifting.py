"""State Lifting Pass."""

from collections.abc import Iterable

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def _get_node_items(block: object) -> Iterable[tuple[str, IRNode]]:
    """Evaluate _get_node_items operation.

    Args:
        block (object): The block parameter.

    Returns:
        object: Result.
    """
    if not hasattr(block, "nodes"):
        return []
    nodes = block.nodes
    if isinstance(nodes, dict):
        return list(nodes.items())
    return [(n.id, n) for n in nodes]


def _lift_node_state(node: IRNode, nid: str, block: object) -> bool:
    """Evaluate _lift_node_state operation.

    Args:
        node (IRNode): The node parameter.
        nid (str): The nid parameter.
        block (object): The block parameter.

    Returns:
        bool: Result.
    """
    if node.op_type == "ReadVariable":
        var_name = node.attributes.get("variable_name", f"var_{nid}")
        node.op_type = "Input"
        node.attributes["name"] = var_name
        return True

    if node.op_type == "AssignVariable":
        var_name = node.attributes.get("variable_name", f"var_{nid}")
        node.op_type = "Output"
        node.attributes["name"] = f"{var_name}_out"
        if hasattr(block, "outputs") and nid not in block.outputs:
            block.outputs.append(nid)
        return True

    return False


def _lift_block(block: object) -> bool:
    """Evaluate _lift_block operation.

    Args:
        block (object): The block parameter.

    Returns:
        bool: Result.
    """
    block_mod = False
    for nid, node in _get_node_items(block):
        block_mod = _lift_node_state(node, nid, block) or block_mod

        for attr_val in node.attributes.values():
            if hasattr(attr_val, "nodes"):
                block_mod = _lift_block(attr_val) or block_mod
    return block_mod


def state_lifting_pass(graph: IRGraph) -> bool:
    """In-place pass to lift state to pure functional inputs.

    Args:
        graph (IRGraph): The graph parameter for the operation.

    Returns:
        bool: A boolean indicating the result of the check.
    """
    return _lift_block(graph)
