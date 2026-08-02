"""Graph scheduling logic for memory-aware and compute-aware execution order."""

from typing import Protocol

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


class CostModel(Protocol):
    """Backend-agnostic Cost Model interface for nodes."""

    def get_memory_cost(self, node: IRNode) -> int:
        """Get the memory cost of a node in bytes.

        Args:
            node (IRNode): The node to evaluate.

        Returns:
            int: The memory cost.
        """
        ...

    def get_compute_cost(self, node: IRNode) -> int:
        """Get the compute cost of a node.

        Args:
            node (IRNode): The node to evaluate.

        Returns:
            int: The compute cost.
        """
        ...


class DefaultCostModel:
    """Default implementation of the cost model."""

    def get_memory_cost(self, node: IRNode) -> int:
        """Calculate memory cost (byte size of output).

        Args:
            node (IRNode): The node.

        Returns:
            int: Memory size in bytes.
        """
        size = 1
        if node.shape_metadata and not node.is_dynamic_shape:
            for dim in node.static_shape:
                size *= max(1, int(dim))

        dtype = node.attributes.get("dtype", "float32")
        sizes = {
            "float64": 8,
            "float32": 4,
            "float16": 2,
            "bfloat16": 2,
            "int64": 8,
            "int32": 4,
            "int16": 2,
            "int8": 1,
            "uint64": 8,
            "uint32": 4,
            "uint16": 2,
            "uint8": 1,
            "bool": 1,
        }
        return size * sizes.get(dtype, 4)

    def get_compute_cost(self, node: IRNode) -> int:
        """Calculate compute cost.

        Args:
            node (IRNode): The node.

        Returns:
            int: The compute cost heuristic.
        """
        if node.op_type in {"MatMul", "Conv2D", "BatchMatMul"}:
            return 1000
        if node.op_type in {"Add", "Sub", "Mul", "Div"}:
            return 10
        return 50


def _build_adjacency_lists(graph: IRGraph) -> tuple[dict[str, list[str]], dict[str, int], dict[str, int]]:
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
    node = graph.nodes[node_id]

    mem_cost = cost_model.get_memory_cost(node)
    mem_freed = 0
    for inp in node.inputs:
        if inp in remaining_uses and remaining_uses[inp] == 1:
            mem_freed += cost_model.get_memory_cost(graph.nodes[inp])

    net_mem = mem_cost - mem_freed

    comp_cost = cost_model.get_compute_cost(node)
    is_compute_heavy = comp_cost > 100

    interleave_penalty = 0
    if is_compute_heavy and last_was_compute_heavy:
        interleave_penalty = 500
    elif not is_compute_heavy and not last_was_compute_heavy:
        interleave_penalty = 100

    return float(net_mem + interleave_penalty)


def _select_best_node(ready_nodes: list[str], graph: IRGraph, cost_model: CostModel, remaining_uses: dict[str, int], last_was_compute_heavy: bool) -> tuple[int, str]:
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
        last_was_compute_heavy = comp_cost > 100

        for inp in node.inputs:
            if inp in remaining_uses:
                remaining_uses[inp] -= 1

        for consumer in consumers[best_node_id]:
            in_degree[consumer] -= 1
            if in_degree[consumer] == 0:
                ready_nodes.append(consumer)

    if len(scheduled_order) != len(graph.nodes):
        return False

    if scheduled_order == list(graph.nodes.keys()):
        return False

    new_nodes = {node_id: graph.nodes[node_id] for node_id in scheduled_order}
    graph.nodes.clear()
    graph.nodes.update(new_nodes)
    return True
