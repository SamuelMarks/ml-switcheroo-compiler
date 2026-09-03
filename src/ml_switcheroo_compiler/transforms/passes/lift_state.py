# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Lift State pass."""

import typing
from collections.abc import Iterable

from ml_switcheroo_compiler.ir.core import IRBlock, IRGraph, IRNode

StateValue = typing.Union[int, float, str, bool, list, tuple, dict, None]


def flatten_state_dict(state_dict: dict[str, typing.Union[StateValue, dict[str, StateValue]]], prefix: str = "") -> dict[str, StateValue]:
    """Flatten a nested state dictionary (like flax.nnx.State) into a flat map.

    Args:
        state_dict: Nested state dict
        prefix: Prefix for keys

    Returns:
        Flattened state map
    """
    flat: dict[str, StateValue] = {}
    for k, v in state_dict.items():
        new_key: str = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(flatten_state_dict(typing.cast(dict[str, typing.Union[StateValue, dict[str, StateValue]]], v), new_key))
        else:
            flat[new_key] = v
    return flat


def unflatten_state_dict(flat_state: dict[str, StateValue]) -> dict[str, typing.Union[StateValue, dict[str, StateValue]]]:
    """Unflatten a state dict back to nested structure.

    Args:
        flat_state: Flattened state map

    Returns:
        Nested state dict
    """
    nested: dict[str, typing.Union[StateValue, dict[str, StateValue]]] = {}
    for k, v in flat_state.items():
        parts: list[str] = k.split(".")
        d: dict[str, typing.Any] = nested  # using typing.Any locally for structural dict construction
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = v
    return nested


def _get_nodes(block: typing.Union[IRGraph, IRBlock]) -> Iterable[IRNode]:
    """Evaluate _get_nodes operation.

    Args:
        block (typing.Union[IRGraph, IRBlock]): The block parameter.

    Returns:
            Iterable[IRNode]: Result.
    """
    nodes: typing.Union[list[IRNode], dict[str, IRNode]] = getattr(block, "nodes", [])
    if isinstance(nodes, dict):
        return nodes.values()
    return nodes


def _lift_node(node: IRNode, block: typing.Union[IRGraph, IRBlock]) -> bool:
    """Evaluate _lift_node operation.

    Args:
        node (IRNode): The node parameter.
        block (typing.Union[IRGraph, IRBlock]): The block parameter.

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

        # VERY IMPORTANT: When integrating state updates into the autodiff gradient tape,
        # we must ensure that the output node inherits a gradient passthrough hook so
        # that reverse-mode AD accurately tracks mutating state backward.
        node.attributes["stop_gradient"] = False
        node.attributes["is_state_update"] = True

        if hasattr(block, "outputs") and node.id not in block.outputs:
            block.outputs.append(node.id)
        return True
    return False


def _lift_block_ir(block: typing.Union[IRGraph, IRBlock]) -> bool:
    """Evaluate _lift_block_ir operation.

    Args:
        block (typing.Union[IRGraph, IRBlock]): The block parameter.

    Returns:
        bool: Result.
    """
    mod: bool = False
    for node in _get_nodes(block):
        mod = _lift_node(node, block) or mod
        for attr_val in node.attributes.values():
            if hasattr(attr_val, "nodes"):
                mod = _lift_block_ir(typing.cast(typing.Union[IRGraph, IRBlock], attr_val)) or mod
    return mod


def lift_state_pass(graph: IRGraph) -> bool:
    """In-place pass to lift implicit state into functional I/O.

    Args:
        graph (IRGraph): The graph parameter.

    Returns:
        bool: Result.
    """
    return _lift_block_ir(graph)
