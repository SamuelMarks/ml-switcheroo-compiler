# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Dead Code Elimination pass."""

import os
from collections.abc import Set

import yaml

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.pass_manager import DAGTopologicalSorter
from ml_switcheroo_compiler.transforms.passes.config_models import BehaviorDescriptorsConfig

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "pass_config", "behavior_descriptors.yaml")
with open(_CONFIG_PATH) as f:
    _config = BehaviorDescriptorsConfig(**yaml.safe_load(f))
SIDE_EFFECT_OPS: set[str] = set(_config.side_effect_ops)

SUBGRAPH_ATTR_KEYS: tuple[str, ...] = (
    "body",
    "subgraph",
    "true_branch",
    "false_branch",
    "cond_branch",
    "loop_body",
)


def _node_has_side_effects(node: IRNode) -> bool:
    """Check if a node or any of its nested subgraphs has side effects.

    Args:
        node (IRNode): The IR node to evaluate.

    Returns:
        bool: True if the node or its subgraphs exhibit side effects.
    """
    if node.op_type in SIDE_EFFECT_OPS:
        return True

    for sub in getattr(node, "subgraphs", {}).values():
        if isinstance(sub, IRGraph):
            if any(_node_has_side_effects(sn) for sn in sub.nodes.values()):
                return True

    for attr_name in SUBGRAPH_ATTR_KEYS:
        sub = node.attributes.get(attr_name)
        if isinstance(sub, IRGraph):
            if any(_node_has_side_effects(sn) for sn in sub.nodes.values()):
                return True
    return False


def _find_side_effect_nodes(graph: IRGraph) -> set[str]:
    """Find all nodes in graph that perform side-effecting operations.

    Args:
        graph (IRGraph): Target computation graph.

    Returns:
        set[str]: Set of node IDs with side effects.
    """
    return {node.id for node in graph.nodes.values() if _node_has_side_effects(node)}


def _prune_nested_subgraphs(node: IRNode) -> bool:
    """Recursively eliminate dead code and prune unused outputs in nested subgraphs.

    Args:
        node (IRNode): Parent IR node containing subgraphs.

    Returns:
        bool: True if any nested subgraph was modified.
    """
    modified = False
    for sub in getattr(node, "subgraphs", {}).values():
        if isinstance(sub, IRGraph):
            sub_mod = dce_pass(sub)
            if sub_mod:
                modified = True
    for attr_name in SUBGRAPH_ATTR_KEYS:
        sub = node.attributes.get(attr_name)
        if isinstance(sub, IRGraph):
            sub_mod = dce_pass(sub)
            if sub_mod:
                modified = True
    return modified


def _build_reachable_set(graph: IRGraph, initial_reachable: Set[str]) -> set[str]:
    """Build the set of reachable nodes via backward traversal from initial roots.

    Args:
        graph (IRGraph): The graph parameter.
        initial_reachable (Set[str]): Initial reachable node IDs (e.g. outputs, side effects).

    Returns:
        set[str]: Fully expanded set of reachable node IDs.
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

    Prunes unused subgraph outputs, side-effect-free loops, and orphaned constants.

    Args:
        graph (IRGraph): The graph parameter.

    Returns:
        bool: True if the graph or any nested subgraphs were modified.
    """
    subgraphs_modified = False
    for node in graph.nodes.values():
        if _prune_nested_subgraphs(node):
            subgraphs_modified = True

    initial_reachable = set(getattr(graph, "outputs", ())) | _find_side_effect_nodes(graph)
    reachable = _build_reachable_set(graph, initial_reachable)

    nodes_to_remove: list[str] = []
    for nid in graph.nodes:
        if nid not in reachable:
            nodes_to_remove.append(nid)

    for nid in nodes_to_remove:
        del graph.nodes[nid]

    # Clean any dead references in graph outputs if node was pruned
    if hasattr(graph, "outputs"):
        new_outputs = [out for out in graph.outputs if out in graph.nodes]
        if new_outputs != graph.outputs:
            graph.outputs = new_outputs
            subgraphs_modified = True

    return len(nodes_to_remove) > 0 or subgraphs_modified
