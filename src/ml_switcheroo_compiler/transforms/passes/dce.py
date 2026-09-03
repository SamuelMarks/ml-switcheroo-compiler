# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Module dce.py."""

"""Dead Code Elimination pass."""

import os

import yaml

from ml_switcheroo_compiler.ir.core import IRGraph
from ml_switcheroo_compiler.transforms.pass_manager import DAGTopologicalSorter
from ml_switcheroo_compiler.transforms.passes.config_models import BehaviorDescriptorsConfig

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "pass_config", "behavior_descriptors.yaml")
with open(_CONFIG_PATH) as f:
    _config = BehaviorDescriptorsConfig(**yaml.safe_load(f))
SIDE_EFFECT_OPS: set[str] = set(_config.side_effect_ops)


def _find_side_effect_nodes(graph: IRGraph) -> set[str]:
    """Evaluate _find_side_effect_nodes operation.

    Args:
        graph (IRGraph): The graph parameter.

    Returns:
        set: Result.
    """
    return {node.id for node in graph.nodes.values() if node.op_type in SIDE_EFFECT_OPS}


def _build_reachable_set(graph: IRGraph, initial_reachable: set[str]) -> set[str]:
    """Evaluate _build_reachable_set operation.

    Args:
        graph (IRGraph): The graph parameter.
        initial_reachable (set): The initial_reachable parameter.

    Returns:
        set: Result.
    """
    reachable = set(initial_reachable)
    sorted_nodes = DAGTopologicalSorter.sort(graph)
    for node in reversed(sorted_nodes):
        if node.id in reachable:
            for inp in node.inputs:
                reachable.add(inp)
    return reachable


def dce_pass(graph: IRGraph) -> bool:
    """In-place Dead Code Elimination (DCE).

    Args:
        graph (IRGraph): The graph parameter.

    Returns:
        bool: Result.
    """
    initial_reachable = set(graph.outputs) | _find_side_effect_nodes(graph)
    reachable = _build_reachable_set(graph, initial_reachable)

    nodes_to_remove = []
    for nid in graph.nodes:
        if nid not in reachable:
            nodes_to_remove.append(nid)

    for nid in nodes_to_remove:
        del graph.nodes[nid]

    return len(nodes_to_remove) > 0
