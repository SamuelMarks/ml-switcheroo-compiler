"""State Lowering pass."""

from ml_switcheroo_compiler.ir.core import IRGraph
from ml_switcheroo_compiler.transforms.pass_manager import DAGTopologicalSorter


def state_lowering_pass(graph: IRGraph) -> bool:
    """In-place State Lowering pass.

    Lowers functional I/O bounds into explicit state assignments (e.g. ReadVariable/AssignVariable).

    Args:
        graph (IRGraph): The input graph to mutate.

    Returns:
        bool: True if the graph was modified, False otherwise.
    """
    modified = False

    sorted_nodes = DAGTopologicalSorter.sort(graph)

    for node in sorted_nodes:
        if node.op_type == "Input" and node.attributes.get("is_state", False):
            node.op_type = "ReadVariable"
            node.attributes["variable_name"] = node.attributes.get("name", node.id)
            if "name" in node.attributes:
                del node.attributes["name"]
            modified = True
        elif node.op_type == "Output" and node.attributes.get("is_state", False):
            node.op_type = "AssignVariable"
            node.attributes["variable_name"] = node.attributes.get("name", node.id)
            if "name" in node.attributes:
                del node.attributes["name"]
            modified = True

    return modified
