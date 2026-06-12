"""Lift State pass."""

from ml_switcheroo.ir.core import IRGraph
from ml_switcheroo.transforms.pass_manager import DAGTopologicalSorter


def lift_state_pass(graph: IRGraph) -> bool:
    """In-place pass to lift implicit state into functional I/O.

    Transforms stateful operations (like reading/writing variables)
    into additional inputs and outputs of the graph.

    Args:
        graph: The IR graph.

    Returns:
        bool: True if modified.
    """
    modified = False

    # Very basic placeholder logic for lifting state:
    # Look for nodes like "ReadVariable", convert them to "Input"
    # Look for "AssignVariable", convert them to "Output" and wire them up.

    sorted_nodes = DAGTopologicalSorter.sort(graph)
    state_inputs = []
    state_outputs = []

    for node in sorted_nodes:
        if node.op_type == "ReadVariable":
            node.op_type = "Input"
            modified = True
            state_inputs.append(node.id)

        elif node.op_type == "AssignVariable":
            node.op_type = "Output"
            modified = True
            state_outputs.append(node.id)

    if state_outputs:
        for out in state_outputs:
            if out not in graph.outputs:
                graph.outputs.append(out)
                modified = True

    return modified
