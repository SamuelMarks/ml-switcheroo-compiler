# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Lift State pass."""

from collections.abc import Iterable
from typing import Any

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
    nested: dict[str, Any] = {}
    for k, v in flat_state.items():
        parts = k.split(".")
        d = nested
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = v
    return nested


def _get_nodes(block: Any) -> Iterable[IRNode]:
    """Evaluate _get_nodes operation.

    Args:
        block (Any): The block parameter.

    Returns: Any: Result.
    """
    nodes = getattr(block, "nodes", [])
    if isinstance(nodes, dict):
        return nodes.values()
    return nodes


def _lift_node(node: IRNode, block: Any) -> bool:
    """Evaluate _lift_node operation.

    Args:
        node (IRNode): The node parameter.
        block (Any): The block parameter.

    Returns:
        bool: Result.
    """
    if node.op_type == "ReadVariable":
        node.op_type = "Input"
        return True
    if node.op_type in ("AssignVariable", "Assign"):
        node.op_type = "Output"
        if len(node.inputs) > 1:
            # For Assign(var, value), we only want to output the new value
            node.inputs = [node.inputs[1]]
        if hasattr(block, "outputs") and node.id not in block.outputs:
            block.outputs.append(node.id)
        return True
    return False


def _lift_block_ir(block: Any) -> bool:
    """Evaluate _lift_block_ir operation.

    Args:
        block (Any): The block parameter.

    Returns:
        bool: Result.
    """
    mod = False
    for node in _get_nodes(block):
        mod = _lift_node(node, block) or mod
        for attr_val in node.attributes.values():
            if hasattr(attr_val, "nodes"):
                mod = _lift_block_ir(attr_val) or mod
    return mod


def lift_state_pass(graph: IRGraph) -> bool:
    """In-place pass to lift implicit state into functional I/O.

    Args:
        graph (IRGraph): The graph parameter.

    Returns:
        bool: Result.
    """
    return _lift_block_ir(graph)
