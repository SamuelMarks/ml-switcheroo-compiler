"""Lift State pass."""

from typing import Any
from collections.abc import Iterable

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


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


def _get_nodes(block: object) -> Iterable[IRNode]:
    nodes = getattr(block, "nodes", [])
    if isinstance(nodes, dict):
        return nodes.values()
    return nodes


def _lift_node(node: IRNode, block: object) -> bool:
    if node.op_type == "ReadVariable":
        node.op_type = "Input"
        return True
    if node.op_type == "AssignVariable":
        node.op_type = "Output"
        if hasattr(block, "outputs") and node.id not in block.outputs:
            block.outputs.append(node.id)
        return True
    return False


def _lift_block_ir(block: object) -> bool:
    mod = False
    for node in _get_nodes(block):
        mod = _lift_node(node, block) or mod
        for attr_val in node.attributes.values():
            if hasattr(attr_val, "nodes"):
                mod = _lift_block_ir(attr_val) or mod
    return mod


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
    return _lift_block_ir(graph)
