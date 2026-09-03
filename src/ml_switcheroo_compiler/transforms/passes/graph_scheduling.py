"""Module graph_scheduling.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Graph scheduling logic for memory-aware and compute-aware execution order."""


from typing import Protocol

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


class CostModel(Protocol):
    """Backend-agnostic Cost Model interface for nodes."""

    @property
    def compute_heavy_threshold(self) -> int:
        """Threshold for heavy compute cost."""
        ...

    @property
    def heavy_interleave_penalty(self) -> int:
        """Penalty for sequential heavy nodes."""
        ...

    @property
    def light_interleave_penalty(self) -> int:
        """Penalty for sequential light nodes."""
        ...

    def get_memory_cost(self, node: IRNode) -> int | str:
        """Get the memory cost of a node in bytes.

        Args:
            node (IRNode): The node parameter.
            int | str: Result.
        """
        ...

    def get_compute_cost(self, node: IRNode) -> int:
        """Get the compute cost of a node.

        Args:
            node (IRNode): The node parameter.
            int: Result.
        """
        ...


import os

import yaml

from ml_switcheroo_compiler.transforms.passes.config_models import CostModelConfig


class DefaultCostModel:
    """Default implementation of the cost model."""

    def __init__(self) -> None:
        """Initialize default cost model.

        Returns:
            None: Result.
        """
        yaml_path = os.path.join(os.path.dirname(__file__), "cost_models.yaml")
        with open(yaml_path) as f:
            self.config = CostModelConfig(**yaml.safe_load(f))

    @property
    def compute_heavy_threshold(self) -> int:
        """Threshold for heavy compute cost."""
        return self.config.compute_heavy_threshold

    @property
    def heavy_interleave_penalty(self) -> int:
        """Penalty for sequential heavy nodes."""
        return self.config.heavy_interleave_penalty

    @property
    def light_interleave_penalty(self) -> int:
        """Penalty for sequential light nodes."""
        return self.config.light_interleave_penalty

    def get_memory_cost(self, node: IRNode) -> int | str:
        """Calculate memory cost (byte size of output).

        Args:
            node (IRNode): The node.

        Returns:
            int | str: Memory size in bytes.
        """
        dtype = node.attributes.get("dtype", "float32")
        sizes = self.config.memory_sizes
        dtype_size = sizes.get(dtype, 4)

        shape = getattr(node, "shape_metadata", None)
        if shape is None:
            return dtype_size

        is_dynamic = getattr(node, "is_dynamic_shape", False)
        has_symbolic_dim = any(isinstance(d, str) for d in shape)

        if not is_dynamic and not has_symbolic_dim:
            size = 1
            for dim in shape:
                size *= max(1, int(dim))
            return size * dtype_size

        dims = [str(d) for d in shape]
        if dims:
            symbolic_math = " * ".join(dims) + f" * {dtype_size}"
            return symbolic_math
        return dtype_size

    def get_compute_cost(self, node: IRNode) -> int:
        """Calculate compute cost.

        Args:
            node (IRNode): The node.

        Returns:
            int: The compute cost heuristic.
        """
        costs = self.config.compute_costs
        if node.op_type in costs.heavy_ops:
            return costs.heavy_cost
        if node.op_type in costs.light_ops:
            return costs.light_cost
        return costs.default_cost


def _build_adjacency_lists(graph: IRGraph) -> tuple[dict[str, list[str]], dict[str, int], dict[str, int]]:
    """Build adjacency lists.

    Args:
        graph (IRGraph): The graph.

    Returns:
        tuple: (consumers, remaining_uses, in_degree)
    """
    consumers: dict[str, list[str]] = {node_id: [] for node_id in graph.nodes}
    remaining_uses = {node_id: 0 for node_id in graph.nodes}
    in_degree = {node_id: 0 for node_id in graph.nodes}
    for node_id, node in graph.nodes.items():
        for inp in node.inputs:
            if inp in graph.nodes:
                consumers[inp].append(node_id)
                remaining_uses[inp] += 1
                in_degree[node_id] += 1
    return consumers, remaining_uses, in_degree


def _score_node(node_id: str, graph: IRGraph, cost_model: CostModel, remaining_uses: dict[str, int], last_was_compute_heavy: bool) -> float:
    """Score a node for scheduling.

    Args:
        node_id (str): The node ID.
        graph (IRGraph): The graph.
        cost_model (CostModel): The cost model.
        remaining_uses (dict): Remaining uses.
        last_was_compute_heavy (bool): True if previous node was compute heavy.

    Returns:
        float: The node score.
    """
    node = graph.nodes[node_id]
    mem_cost = cost_model.get_memory_cost(node)
    mem_freed = 0
    for inp in node.inputs:
        if inp in remaining_uses and remaining_uses[inp] == 1:
            mem_freed += cost_model.get_memory_cost(graph.nodes[inp])
    net_mem = mem_cost - mem_freed
    comp_cost = cost_model.get_compute_cost(node)
    is_compute_heavy = comp_cost > cost_model.compute_heavy_threshold
    interleave_penalty = 0
    if is_compute_heavy and last_was_compute_heavy:
        interleave_penalty = cost_model.heavy_interleave_penalty
    elif not is_compute_heavy and not last_was_compute_heavy:
        interleave_penalty = cost_model.light_interleave_penalty
    return float(net_mem + interleave_penalty)


def _select_best_node(ready_nodes: list[str], graph: IRGraph, cost_model: CostModel, remaining_uses: dict[str, int], last_was_compute_heavy: bool) -> tuple[int, str]:
    """Select the best node.

    Args:
        ready_nodes (list[str]): List of ready nodes.
        graph (IRGraph): The graph.
        cost_model (CostModel): The cost model.
        remaining_uses (dict): Remaining uses.
        last_was_compute_heavy (bool): True if previous node was compute heavy.

    Returns:
        tuple[int, str]: Index and node ID of the best node.
    """
    best_node_id = None
    best_score = float("inf")
    best_idx = -1
    for i, node_id in enumerate(ready_nodes):
        score = _score_node(node_id, graph, cost_model, remaining_uses, last_was_compute_heavy)
        if score < best_score or (score == best_score and (best_node_id is None or node_id < best_node_id)):
            best_score = score
            best_node_id = node_id
            best_idx = i
    assert best_node_id is not None
    return best_idx, best_node_id


def _update_degrees_and_uses(node: IRNode, best_node_id: str, consumers: dict[str, list[str]], remaining_uses: dict[str, int], in_degree: dict[str, int], ready_nodes: list[str]) -> None:
    """Update adjacency state after scheduling.

    Args:
        node (IRNode): The node.
        best_node_id (str): The node ID.
        consumers (dict): Consumers map.
        remaining_uses (dict): Remaining uses map.
        in_degree (dict): In-degree map.
        ready_nodes (list): Ready nodes list.
    """
    for inp in node.inputs:
        if inp in remaining_uses:
            remaining_uses[inp] -= 1
    for consumer in consumers[best_node_id]:
        in_degree[consumer] -= 1
        if in_degree[consumer] == 0:
            ready_nodes.append(consumer)


def graph_scheduling_pass(graph: IRGraph) -> bool:
    """Reorder graph nodes for optimal execution schedule.

    Implements memory-aware topological sorting and compute-aware scheduling
    by prioritizing paths that free large memory buffers and interleaving
    compute/memory bound ops.

    Args:
        graph (IRGraph): The input graph to mutate.

    Returns:
        bool: True if the graph was modified, False otherwise.
    """
    cost_model = DefaultCostModel()
    consumers, remaining_uses, in_degree = _build_adjacency_lists(graph)
    ready_nodes = [node_id for node_id, deg in in_degree.items() if deg == 0]
    scheduled_order = []
    last_was_compute_heavy = False
    while ready_nodes:
        best_idx, best_node_id = _select_best_node(ready_nodes, graph, cost_model, remaining_uses, last_was_compute_heavy)
        ready_nodes.pop(best_idx)
        scheduled_order.append(best_node_id)
        node = graph.nodes[best_node_id]
        comp_cost = cost_model.get_compute_cost(node)
        last_was_compute_heavy = comp_cost > cost_model.compute_heavy_threshold
        _update_degrees_and_uses(node, best_node_id, consumers, remaining_uses, in_degree, ready_nodes)
    if len(scheduled_order) != len(graph.nodes):
        return False
    if scheduled_order == list(graph.nodes.keys()):
        return False
    new_nodes = {node_id: graph.nodes[node_id] for node_id in scheduled_order}
    graph.nodes.clear()
    graph.nodes.update(new_nodes)
    return True
