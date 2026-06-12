"""Lift State pass."""

from typing import Any

from ml_switcheroo_compiler.ir.core import IRGraph
from ml_switcheroo_compiler.transforms.pass_manager import DAGTopologicalSorter


def flatten_state_dict(state_dict: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested state dictionary (like flax.nnx.State) into a flat map.

    Args:
        state_dict (Dict[str, Any]): Nested state dict
        prefix (str): Prefix for keys

    Returns:
        Dict[str, Any]: Flattened state map
    """
    flat = {}
    for k, v in state_dict.items():
        new_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(flatten_state_dict(v, new_key))
        else:
            flat[new_key] = v
    return flat


def unflatten_state_dict(flat_state: dict[str, Any]) -> dict[str, Any]:
    """Unflatten a state dict back to nested structure.

    Args:
        flat_state (Dict[str, Any]): Flattened state map

    Returns:
        Dict[str, Any]: Nested state dict
    """
    nested = {}
    for k, v in flat_state.items():
        parts = k.split(".")
        d = nested
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = v
    return nested


def lift_state_pass(graph: IRGraph) -> bool:
    """In-place pass to lift implicit state into functional I/O.

    Transforms stateful operations (like reading/writing variables)
    into additional inputs and outputs of the graph

    graph: The IR graph

    Returns:
    bool: True if modified

    Args:
    graph (IRGraph): Argument graph
    """
    modified = False

    # Very basic placeholder logic for lifting state:
    # Look for nodes like "ReadVariable", convert them to "Input"
    # Look for "AssignVariable", convert them to "Output" and wire them up

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
