"""Buffer Allocation pass for edge execution."""

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.pass_manager import DAGTopologicalSorter


def _get_dtype_size(dtype_str: str) -> int:
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
    return sizes.get(dtype_str, 4)


def _get_node_byte_size(node: IRNode) -> int:
    dtype = node.attributes.get("dtype", "float32")
    dtype_size = _get_dtype_size(dtype)
    if node.shape_metadata is None or node.is_dynamic_shape:
        return dtype_size

    elements = 1
    for dim in node.static_shape:
        elements *= max(1, int(dim))
    return elements * dtype_size


IN_PLACE_SAFE_OPS: set[str] = {
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Relu",
    "Exp",
    "Log",
    "Tanh",
    "Sigmoid",
    "Cast",
    "Reshape",
    "Squeeze",
    "Unsqueeze",
    "Sqrt",
    "Abs",
    "Maximum",
    "Minimum",
}


class GreedyOffsetAllocator:
    """A simple offset allocator using linear scan over active intervals."""

    def __init__(self) -> None:
        """Initialize the allocator."""
        self.active_allocations: list[tuple[int, int, int]] = []

    def allocate(self, size: int, current_time: int, end_time: int) -> int:
        """Allocate a contiguous block of size `size`.

        Args:
            size (int): The size of the allocation.
            current_time (int): The current timestep.
            end_time (int): The timestep when the allocation expires.

        Returns:
            int: The allocated offset.
        """
        self.active_allocations = [alloc for alloc in self.active_allocations if alloc[2] > current_time]

        self.active_allocations.sort(key=lambda x: x[0])

        current_offset = 0
        for offset, alloc_size, _ in self.active_allocations:
            if current_offset + size <= offset:
                break
            current_offset = max(current_offset, offset + alloc_size)

        self.active_allocations.append((current_offset, size, end_time))
        return current_offset

    def allocate_at(self, offset: int, size: int, end_time: int) -> None:
        """Force allocation at a specific offset (e.g. for in-place).

        Args:
            offset (int): The offset to allocate at.
            size (int): The size of the allocation.
            end_time (int): The timestep when the allocation expires.
        """
        self.active_allocations.append((offset, size, end_time))


def _compute_liveness(graph: IRGraph, sorted_nodes: list[IRNode]) -> dict[str, int]:
    """Compute liveness intervals for nodes.

    Args:
        graph: The IR graph.
        sorted_nodes: Nodes in topological order.

    Returns:
        dict[str, int]: A map from node ID to its last use timestep.
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
    """In-place Buffer Allocation pass.

    Assigns memory buffers (offsets and sizes) to nodes for lower-level execution
    targets like WASM and WebGPU, implementing liveness analysis and memory pooling.

    Args:
        graph (IRGraph): The input graph to mutate.

    Returns:
        bool: True if the graph was modified, False otherwise.
    """
    sorted_nodes = DAGTopologicalSorter.sort(graph)
    if not sorted_nodes:
        return False

    last_use = _compute_liveness(graph, sorted_nodes)
    allocator = GreedyOffsetAllocator()
    modified = False

    for i, node in enumerate(sorted_nodes):
        size = _get_node_byte_size(node)
        reused_offset = _try_reuse_buffer(node, graph, size, i, last_use)

        if reused_offset >= 0:
            assigned_offset = reused_offset
            allocator.allocate_at(assigned_offset, size, last_use[node.id])
        else:
            assigned_offset = allocator.allocate(size, i, last_use[node.id])

        if node.attributes.get("buffer_offset") != assigned_offset:
            node.attributes["buffer_offset"] = assigned_offset
            node.attributes["buffer_size"] = size
            modified = True

    return modified
