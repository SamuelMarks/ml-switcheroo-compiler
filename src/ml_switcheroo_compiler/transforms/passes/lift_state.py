"""Unified state lifting pass utilities delegating to StateLiftingPass."""

from __future__ import annotations

import typing
from collections.abc import Iterable, Mapping

from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.core.shape import TensorOrScalar, TypedVariadicCallable
from ml_switcheroo_compiler.ir.core import IRGraph
from ml_switcheroo_compiler.transforms.passes.state_lifting import (
    PureStateCallableProtocol,
    StateEntry,
    StateLiftingPass,
    _get_node_items,
    _lift_block,
    _lift_node_state,
    state_lifting_pass,
)

StateValue = StateEntry


def _is_nested_state_dict(
    val: StateValue | dict[str, StateValue | dict[str, StateValue]] | None,
) -> typing.TypeGuard[dict[str, StateValue | dict[str, StateValue]]]:
    """Type guard narrowing `val` to a nested state dictionary.

    Args:
        val (StateValue | dict[str, StateValue | dict[str, StateValue]] | None): Candidate value.

    Returns:
        bool: True if `val` is a dictionary.
    """
    return isinstance(val, dict)


def flatten_state_dict(
    state_dict: Mapping[str, StateValue],
    prefix: str = "",
) -> dict[str, StateValue]:
    """Flatten a nested state dictionary into a dot-delimited flat map.

    Args:
        state_dict (Mapping[str, StateValue]): Nested state dictionary.
        prefix (str): Key prefix for recursion.

    Returns:
        dict[str, StateValue]: Flattened state map.
    """
    flat: dict[str, StateValue] = {}
    for k, v in state_dict.items():
        new_key: str = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            sub_map: dict[str, StateValue] = {str(sk): sv for sk, sv in v.items()}
            flat.update(flatten_state_dict(sub_map, new_key))
        else:
            flat[new_key] = v
    return flat


def unflatten_state_dict(
    flat_state: dict[str, StateValue],
) -> dict[str, StateValue | dict[str, StateValue]]:
    """Unflatten a dot-delimited state dictionary back into a nested structure.

    Args:
        flat_state (dict[str, StateValue]): Flattened state map.

    Returns:
        dict[str, StateValue | dict[str, StateValue]]: Nested state dictionary.
    """
    nested: dict[str, StateValue | dict[str, StateValue]] = {}
    for k, v in flat_state.items():
        parts: list[str] = k.split(".")
        d: dict[str, StateValue | dict[str, StateValue]] = nested
        for part in parts[:-1]:
            existing = d.get(part)
            if _is_nested_state_dict(existing):
                d = existing
            else:
                sub_dict: dict[str, StateValue | dict[str, StateValue]] = {}
                d[part] = sub_dict
                d = sub_dict
        d[parts[-1]] = v
    return nested


def _get_nodes(block: LogicalGraph) -> Iterable[LogicalNode]:
    """Extract all LogicalNode instances from a block or graph.

    Args:
        block (LogicalGraph): The block or graph to inspect.

    Returns:
        Iterable[LogicalNode]: Sequence of LogicalNode instances.
    """
    return [node for _, node in _get_node_items(block)]


def _lift_node(node: LogicalNode, block: LogicalGraph) -> bool:
    """Lift a single ReadVariable/AssignVariable/Assign node in place.

    Args:
        node (LogicalNode): Target node.
        block (LogicalGraph): Containing block or graph.

    Returns:
        bool: True if the node was modified.
    """
    return _lift_node_state(node, node.id, block)


def _lift_block_ir(block: LogicalGraph) -> bool:
    """Recursively lift stateful operations across a block and its subgraphs.

    Args:
        block (LogicalGraph): Target block or graph.

    Returns:
        bool: True if any node was lifted.
    """
    return _lift_block(block)


def lift_state_pass(graph: IRGraph) -> bool:
    """In-place pass to lift implicit state into functional I/O.

    Args:
        graph (IRGraph): Target IRGraph.

    Returns:
        bool: True if the graph was modified.
    """
    return state_lifting_pass(graph)


def lift_module_state(container: TensorOrScalar) -> tuple[dict[str, StateValue], IRGraph]:
    """Lift a variable container into pure function parameters and an IRGraph.

    Args:
        container (TensorOrScalar): Stateful variable container.

    Returns:
        tuple[dict[str, StateValue], IRGraph]: Extracted state map and functional IRGraph.
    """
    return StateLiftingPass().lift(container)


def lift_state(
    container: TypedVariadicCallable[TensorOrScalar],
) -> tuple[dict[str, StateValue], PureStateCallableProtocol]:
    """Functionalize a callable variable container into pure signature (state, *args) -> (out, new_state).

    Args:
        container (TypedVariadicCallable[TensorOrScalar]): Callable stateful container.

    Returns:
        tuple[dict[str, StateValue], PureStateCallableProtocol]: Initial state dict and pure callable.
    """
    return StateLiftingPass().functionalize(container)


__all__ = [
    "StateLiftingPass",
    "StateValue",
    "_get_nodes",
    "_lift_block_ir",
    "_lift_node",
    "flatten_state_dict",
    "lift_module_state",
    "lift_state",
    "lift_state_pass",
    "unflatten_state_dict",
]
