# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Module buffer_allocation.py."""

"""Buffer Allocation pass for edge execution."""

import os

import yaml

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.pass_manager import DAGTopologicalSorter
from ml_switcheroo_compiler.transforms.passes.config_models import OptimizationHeuristicsConfig

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "pass_config", "optimization_heuristics.yaml")
with open(_CONFIG_PATH) as f:
    _config = OptimizationHeuristicsConfig(**yaml.safe_load(f))
IN_PLACE_SAFE_OPS: set[str] = set(_config.in_place_safe_ops)


def _get_node_byte_size(node: IRNode):
    """Calculate the byte size of a node's output tensor, supporting symbolic shapes.

    Args:
        node (IRNode): The IR node.

    Returns:
        int | str: The size in bytes (int) or a symbolic expression (str).
    """
    from ml_switcheroo_compiler.transforms.passes.graph_scheduling import DefaultCostModel

    cost_model = DefaultCostModel()
    cost = cost_model.get_memory_cost(node)
    return cost if isinstance(cost, (int, str)) else str(cost)


class GreedyOffsetAllocator:
    """Greedy offset allocator supporting dynamic symbolic shapes."""

    def __init__(self) -> None:
        """Initialize allocator."""
        self.current_static_offset = 0
        self.active_blocks: list[tuple[int, int, int]] = []
        self.dynamic_blocks: list[tuple[str, str, int]] = []

    def allocate_static(self, size: int, current_time: int, expire_time: int) -> int:
        """Allocate static size."""
        self.active_blocks = [b for b in self.active_blocks if b[2] >= current_time]
        self.active_blocks.sort(key=lambda x: x[0])

        offset = 0
        for block in self.active_blocks:
            if block[0] - offset >= size:
                break
            offset = max(offset, block[1])

        self.active_blocks.append((offset, offset + size, expire_time))
        return offset

    def allocate(self, size: int, current_time: int, expire_time: int) -> int:
        """Allocate static size compat."""
        return self.allocate_static(size, current_time, expire_time)

    def allocate_dynamic(self, symbolic_math: str, current_time: int, expire_time: int, var_name: str) -> str:
        """Allocate dynamic size."""
        self.dynamic_blocks = [b for b in self.dynamic_blocks if b[2] >= current_time]
        self.dynamic_blocks.append((var_name, symbolic_math, expire_time))
        return f"offset_{var_name}"

    def allocate_at(self, offset: int, size: int, expire_time: int) -> None:
        """Allocate at specific static offset."""
        self.active_blocks.append((offset, offset + size, expire_time))


def _compute_liveness(graph: IRGraph, sorted_nodes: list[IRNode]) -> dict[str, int]:
    """Evaluate _compute_liveness operation.

    Args:
        graph (IRGraph): The graph parameter.
        sorted_nodes (list): The sorted_nodes parameter.

    Returns:
        dict: Result.
    """
    last_use: dict[str, int] = {}
    for i, node in enumerate(sorted_nodes):
        last_use[node.id] = i

    for i, node in enumerate(sorted_nodes):
        for inp in node.inputs:
            if inp in last_use:
                last_use[inp] = max(last_use[inp], i)

    end_time = len(sorted_nodes)
    for out_id in graph.outputs:
        if out_id in last_use:
            last_use[out_id] = end_time

    return last_use


def _try_reuse_buffer(node: IRNode, graph: IRGraph, size: int, i: int, last_use: dict[str, int]) -> int:
    """Attempt to reuse an input's buffer for the node's output.

    Args:
        node: The node to allocate buffer for.
        graph: The IR graph.
        size: Required buffer size.
        i: Current timestep.
        last_use: Map of node last uses.

    Returns:
        int: The reused offset or -1 if no buffer can be reused.
    """
    if node.op_type not in IN_PLACE_SAFE_OPS:
        return -1

    for inp_id in node.inputs:
        if inp_id in graph.nodes:
            inp_node = graph.nodes[inp_id]
            if last_use.get(inp_id, -1) == i:
                inp_offset = inp_node.attributes.get("buffer_offset")
                inp_size = inp_node.attributes.get("buffer_size", 0)
                if inp_offset is not None and int(inp_size) >= size:
                    return int(inp_offset)
    return -1


def buffer_allocation_pass(graph: IRGraph) -> bool:
    """In-place Buffer Allocation pass with dynamic shape support."""
    sorted_nodes = DAGTopologicalSorter.sort(graph)
    if not sorted_nodes:
        return False

    last_use = _compute_liveness(graph, sorted_nodes)
    allocator = GreedyOffsetAllocator()
    modified = False

    # Store dynamic schema configuration
    graph.attributes = getattr(graph, "attributes", {})
    graph.attributes["dynamic_memory_schema"] = {"dynamic_offsets": []}

    for i, node in enumerate(sorted_nodes):
        size = _get_node_byte_size(node)

        if isinstance(size, str):  # Dynamic
            var_name = getattr(node, "id", f"node_{i}")
            assigned_offset = allocator.allocate_dynamic(size, i, last_use.get(node.id, i), var_name)

            # Record the dynamic offset computation request in the graph attributes
            graph.attributes["dynamic_memory_schema"]["dynamic_offsets"].append({"var_name": var_name, "symbolic_math": size, "node_id": node.id})

            if node.attributes.get("buffer_offset_symbolic") != assigned_offset:
                node.attributes["buffer_offset_symbolic"] = assigned_offset
                node.attributes["buffer_size_symbolic"] = size
                node.attributes["buffer_id"] = 0
                modified = True
        else:  # Static
            reused_offset = _try_reuse_buffer(node, graph, size, i, last_use)
            if reused_offset >= 0:
                static_assigned_offset = reused_offset
                allocator.allocate_at(static_assigned_offset, size, last_use.get(node.id, i))
            else:
                static_assigned_offset = allocator.allocate_static(size, i, last_use.get(node.id, i))

            if node.attributes.get("buffer_offset") != static_assigned_offset:
                node.attributes["buffer_offset"] = static_assigned_offset
                node.attributes["buffer_size"] = size
                node.attributes["buffer_id"] = 0
                modified = True

    return modified
