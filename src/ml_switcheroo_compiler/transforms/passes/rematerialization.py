"""Rematerialization pass."""

import os
from typing import Any

import yaml

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.pass_manager import DAGTopologicalSorter


def _load_rules() -> dict[str, Any]:
    """Load rematerialization rules from YAML config.

    Returns:
        dict[str, Any]: The loaded rules.
    """
    yaml_path = os.path.join(os.path.dirname(__file__), "rematerialization_rules.yaml")
    with open(yaml_path) as f:
        from ml_switcheroo_compiler.transforms.passes.config_models import RematerializationRulesConfig

        return RematerializationRulesConfig(**yaml.safe_load(f)).model_dump()


def _estimate_memory(node: IRNode) -> float:
    """Estimate memory footprint of node in bytes.

    Args:
        node (IRNode): The IR node.

    Returns:
        float: Estimated memory in bytes.
    """
    shape = getattr(node, "shape_metadata", None)
    if not shape:
        return 4.0  # default scalar
    if isinstance(shape, (int, float)):
        return 4.0
    bytes_val = 4.0
    for dim in shape:
        bytes_val *= dim
    return bytes_val


def _estimate_compute(node: IRNode, rules: dict[str, Any]) -> float:
    """Estimate FLOPs of a node.

    Args:
        node (IRNode): The IR node.
        rules (dict): The rules dict.

    Returns:
        float: Estimated FLOPs.
    """
    shape = getattr(node, "shape_metadata", None)
    if not shape:
        return 1.0
    flops = 1.0
    if isinstance(shape, (list, tuple)):
        for dim in shape:
            flops *= dim
    if node.op_type in rules.get("high_cost_ops", []):
        flops *= 100  # arbitrary higher cost
    return flops


def _find_target_nodes(nodes: list[IRNode], consumers: dict[str, list[str]], node_indices: dict[str, int], rules: dict[str, Any]) -> list[IRNode]:
    """Find nodes suitable for rematerialization.

    Args:
        nodes (list[IRNode]): Sorted nodes.
        consumers (dict): Mapping from node ID to consumer node IDs.
        node_indices (dict): Mapping from node ID to its topological index.
        rules (dict): The rules dict.

    Returns:
        list[IRNode]: List of nodes to rematerialize.
    """
    target_ops = set(rules.get("target_ops", []))
    min_mem = rules.get("thresholds", {}).get("min_memory_bytes", 1024 * 1024)
    max_ratio = rules.get("thresholds", {}).get("max_compute_to_memory_ratio", 10.0)

    to_remat = []
    for n in nodes:
        if n.op_type in target_ops:
            mem = _estimate_memory(n)
            comp = _estimate_compute(n, rules)
            if mem > min_mem and comp / mem < max_ratio and consumers[n.id]:
                max_dist = max(node_indices[c] for c in consumers[n.id]) - node_indices[n.id]
                if max_dist > 10:
                    to_remat.append(n)
    return to_remat


def rematerialization_pass(graph: IRGraph) -> bool:
    """Drop high-memory/low-compute nodes and inject exact clones for backward pass consumers.

    Args:
        graph (IRGraph): The input IR graph.

    Returns:
        bool: True if the graph was modified, False otherwise.
    """
    modified = False
    rules = _load_rules()

    nodes = DAGTopologicalSorter.sort(graph)
    node_indices = {n.id: i for i, n in enumerate(nodes)}

    consumers: dict[str, list[str]] = {n.id: [] for n in nodes}
    for n in nodes:
        for inp in n.inputs:
            if inp in consumers:
                consumers[inp].append(n.id)

    to_remat = _find_target_nodes(nodes, consumers, node_indices, rules)

    for n in to_remat:
        clone_id = f"{n.id}_remat"
        if clone_id not in graph.nodes:
            new_node = IRNode(id=clone_id, op_type=n.op_type, inputs=n.inputs.copy())
            new_node.attributes = n.attributes.copy()
            new_node.shape_metadata = getattr(n, "shape_metadata", None)
            graph.nodes[clone_id] = new_node
            modified = True

        n.attributes["rematerialize"] = True

        for c_id in consumers[n.id]:
            if node_indices[c_id] - node_indices[n.id] > 10:
                c_node = graph.nodes[c_id]
                # Check if we actually need to change inputs
                if n.id in c_node.inputs:
                    c_node.inputs = [clone_id if inp == n.id else inp for inp in c_node.inputs]
                    modified = True

    return modified
