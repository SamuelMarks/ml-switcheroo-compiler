"""Buffer Allocation pass for edge execution and memory arena planning."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
import os

import yaml

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.pass_manager import DAGTopologicalSorter
from ml_switcheroo_compiler.transforms.passes.config_models import OptimizationHeuristicsConfig

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "pass_config", "optimization_heuristics.yaml")
with open(_CONFIG_PATH) as f:
    _config = OptimizationHeuristicsConfig(**yaml.safe_load(f))
IN_PLACE_SAFE_OPS: set[str] = set(_config.in_place_safe_ops)


def _get_node_byte_size(node: IRNode) -> int | str:
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
        self.current_static_offset: int = 0
        self.active_blocks: list[tuple[int, int, int]] = []
        self.dynamic_blocks: list[tuple[str, str, int]] = []

    def allocate_static(self, size: int, current_time: int, expire_time: int) -> int:
        """Allocate static size.

        Args:
            size (int): Size in bytes.
            current_time (int): Current allocation step.
            expire_time (int): Expiration step.

        Returns:
            int: Allocated byte offset.
        """
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
        """Allocate static size compat.

        Args:
            size (int): Size in bytes.
            current_time (int): Current allocation step.
            expire_time (int): Expiration step.

        Returns:
            int: Allocated byte offset.
        """
        return self.allocate_static(size, current_time, expire_time)

    def allocate_dynamic(self, symbolic_math: str, current_time: int, expire_time: int, var_name: str) -> str:
        """Allocate dynamic size.

        Args:
            symbolic_math (str): Symbolic size expression.
            current_time (int): Current allocation step.
            expire_time (int): Expiration step.
            var_name (str): Variable name.

        Returns:
            str: Dynamic offset identifier.
        """
        self.dynamic_blocks = [b for b in self.dynamic_blocks if b[2] >= current_time]
        self.dynamic_blocks.append((var_name, symbolic_math, expire_time))
        return f"offset_{var_name}"

    def allocate_at(self, offset: int, size: int, expire_time: int) -> None:
        """Allocate at specific static offset.

        Args:
            offset (int): Starting offset in bytes.
            size (int): Size in bytes.
            expire_time (int): Expiration step.
        """
        self.active_blocks.append((offset, offset + size, expire_time))


def _compute_liveness(graph: IRGraph, sorted_nodes: list[IRNode]) -> dict[str, int]:
    """Compute last-use timesteps for all nodes in the graph.

    Args:
        graph (IRGraph): The graph parameter.
        sorted_nodes (list[IRNode]): The sorted_nodes parameter.

    Returns:
        dict[str, int]: Resulting map of node ID to last use timestep.
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
        node (IRNode): The node to allocate buffer for.
        graph (IRGraph): The IR graph.
        size (int): Required buffer size.
        i (int): Current timestep.
        last_use (dict[str, int]): Map of node last uses.

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


class InterferenceGraph:
    """Interference graph representation where vertices are tensors and edges denote overlapping lifetimes."""

    def __init__(self) -> None:
        """Initialize empty interference graph."""
        self.nodes: set[str] = set()
        self.adj: dict[str, set[str]] = {}
        self.sizes: dict[str, int] = {}
        self.intervals: dict[str, tuple[int, int]] = {}

    def add_node(self, node_id: str, size: int, birth: int, death: int) -> None:
        """Add tensor node with lifetime interval and size.

        Args:
            node_id (str): Unique node identifier.
            size (int): Required tensor size in bytes.
            birth (int): Starting timestep of tensor lifetime.
            death (int): Ending timestep of tensor lifetime.
        """
        self.nodes.add(node_id)
        if node_id not in self.adj:
            self.adj[node_id] = set()
        self.sizes[node_id] = size
        self.intervals[node_id] = (birth, death)

    def add_edge(self, u: str, v: str) -> None:
        """Add interference edge between two nodes.

        Args:
            u (str): First node identifier.
            v (str): Second node identifier.
        """
        self.adj[u].add(v)
        self.adj[v].add(u)

    def build_edges_from_intervals(self) -> None:
        """Build interference edges for all pairs of nodes with overlapping lifetimes."""
        node_list = list(self.nodes)
        for i in range(len(node_list)):
            u = node_list[i]
            b_u, d_u = self.intervals[u]
            for j in range(i + 1, len(node_list)):
                v = node_list[j]
                b_v, d_v = self.intervals[v]
                if max(b_u, b_v) <= min(d_u, d_v):
                    self.add_edge(u, v)

    def color_registers(self) -> dict[str, int]:
        """Perform degree-ordered greedy graph coloring (Chaitin-Briggs style heuristic).

        Returns:
            dict[str, int]: Mapping from node ID to color (buffer slot index).
        """
        sorted_nodes = sorted(self.nodes, key=lambda n: len(self.adj.get(n, set())), reverse=True)
        colors: dict[str, int] = {}

        for node in sorted_nodes:
            neighbor_colors = {colors[neighbor] for neighbor in self.adj.get(node, set()) if neighbor in colors}
            c = 0
            while c in neighbor_colors:
                c += 1
            colors[node] = c

        return colors


class InterferenceGraphColoringAllocator:
    """Interference graph register coloring allocator for minimal linear memory arena planning."""

    def __init__(self, alignment: int = 256) -> None:
        """Initialize coloring allocator.

        Args:
            alignment (int): Memory alignment boundary in bytes (default 256 for WebGPU/WASM/SIMD).
        """
        self.alignment: int = alignment

    def build_interference_graph(self, graph: IRGraph, sorted_nodes: list[IRNode]) -> InterferenceGraph:
        """Construct interference graph from topologically sorted IR graph nodes.

        Args:
            graph (IRGraph): The computation graph.
            sorted_nodes (list[IRNode]): Topologically ordered nodes.

        Returns:
            InterferenceGraph: Constructed interference graph.
        """
        birth: dict[str, int] = {}
        for i, node in enumerate(sorted_nodes):
            birth[node.id] = i

        last_use: dict[str, int] = {}
        for i, node in enumerate(sorted_nodes):
            last_use[node.id] = i
            for inp in node.inputs:
                if inp in last_use:
                    last_use[inp] = max(last_use[inp], i)

        end_time = len(sorted_nodes)
        for out_id in getattr(graph, "outputs", []):
            if out_id in last_use:
                last_use[out_id] = end_time

        ig = InterferenceGraph()
        for node in sorted_nodes:
            size_val = _get_node_byte_size(node)
            byte_size = int(size_val) if isinstance(size_val, int) else 1024
            b = birth.get(node.id, 0)
            d = last_use.get(node.id, b)
            ig.add_node(node.id, byte_size, b, d)

        ig.build_edges_from_intervals()
        return ig

    def allocate_colored_buffers(self, graph: IRGraph, sorted_nodes: list[IRNode] | None = None) -> dict[str, dict[str, int]]:
        """Allocate minimal linear memory buffers using graph coloring.

        Args:
            graph (IRGraph): Target computation graph.
            sorted_nodes (typing.Optional[list[IRNode]]): Pre-sorted nodes, if available.

        Returns:
            dict[str, dict[str, int]]: Map of node ID to color, offset, and size.
        """
        if sorted_nodes is None:
            sorted_nodes = DAGTopologicalSorter.sort(graph)
        ig = self.build_interference_graph(graph, sorted_nodes)
        colors = ig.color_registers()

        color_groups: dict[int, list[str]] = {}
        for nid, c in colors.items():
            color_groups.setdefault(c, []).append(nid)

        color_sizes: dict[int, int] = {}
        for c, nids in color_groups.items():
            max_sz = max(ig.sizes[nid] for nid in nids)
            color_sizes[c] = ((max_sz + self.alignment - 1) // self.alignment) * self.alignment

        current_offset = 0
        color_offsets: dict[int, int] = {}
        for c in sorted(color_sizes.keys()):
            color_offsets[c] = current_offset
            current_offset += color_sizes[c]

        allocations: dict[str, dict[str, int]] = {}
        for nid, c in colors.items():
            allocations[nid] = {
                "color": c,
                "offset": color_offsets[c],
                "size": ig.sizes[nid],
            }
        return allocations

    def compute_peak_memory_reduction(self, graph: IRGraph) -> dict[str, float]:
        """Compute memory savings between naive allocation and register-colored allocation.

        Args:
            graph (IRGraph): Target deep neural network architecture.

        Returns:
            dict[str, float]: Comparison metrics (naive_peak_bytes, colored_peak_bytes, reduction_pct).
        """
        sorted_nodes = DAGTopologicalSorter.sort(graph)
        ig = self.build_interference_graph(graph, sorted_nodes)
        colors = ig.color_registers()

        naive_peak = sum(((ig.sizes[nid] + self.alignment - 1) // self.alignment) * self.alignment for nid in ig.nodes)

        color_max_sizes: dict[int, int] = {}
        for nid, c in colors.items():
            color_max_sizes[c] = max(color_max_sizes.get(c, 0), ig.sizes[nid])

        colored_peak = sum(((sz + self.alignment - 1) // self.alignment) * self.alignment for sz in color_max_sizes.values())

        reduction_pct = ((naive_peak - colored_peak) / naive_peak * 100.0) if naive_peak > 0 else 0.0
        return {
            "naive_peak_bytes": float(naive_peak),
            "colored_peak_bytes": float(colored_peak),
            "reduction_pct": float(reduction_pct),
        }


class BufferAllocationPass:
    """Buffer allocation pass implementing lifetime analysis and memory arena calculation."""

    def __init__(self, alignment: int = 16) -> None:
        """Initialize BufferAllocationPass.

        Args:
            alignment (int): Memory alignment boundary in bytes (default: 16 for WebGPU/WASM).
        """
        self.alignment: int = alignment

    def compute_liveness_intervals(self, graph: IRGraph, sorted_nodes: list[IRNode] | None = None) -> dict[str, tuple[int, int]]:
        """Compute the [birth, death] liveness interval for each node's intermediate tensor.

        Args:
            graph (IRGraph): The IR graph.
            sorted_nodes (list[IRNode], optional): Topologically sorted nodes.

        Returns:
            dict[str, tuple[int, int]]: Mapping of node ID to (start_timestep, end_timestep).
        """
        if sorted_nodes is None:
            sorted_nodes = DAGTopologicalSorter.sort(graph)

        birth: dict[str, int] = {}
        last_use: dict[str, int] = {}
        for i, node in enumerate(sorted_nodes):
            birth[node.id] = i
            last_use[node.id] = i

        for i, node in enumerate(sorted_nodes):
            for inp in node.inputs:
                if inp in last_use:
                    last_use[inp] = max(last_use[inp], i)

        end_time = len(sorted_nodes)
        for out_id in graph.outputs:
            if out_id in last_use:
                last_use[out_id] = end_time

        intervals: dict[str, tuple[int, int]] = {}
        for nid, start_t in birth.items():
            intervals[nid] = (start_t, last_use.get(nid, start_t))
        return intervals

    def calculate_arena_offsets(self, graph: IRGraph, alignment: int | None = None) -> dict[str, int]:
        """Calculate minimal static memory arena offsets for WebGPU storage buffers and WASM linear memory.

        Args:
            graph (IRGraph): The IR graph.
            alignment (int, optional): Byte alignment for buffer offsets. Defaults to self.alignment.

        Returns:
            dict[str, int]: Mapping of node ID to assigned byte offset in the linear arena.
        """
        align = alignment if alignment is not None else self.alignment
        sorted_nodes = DAGTopologicalSorter.sort(graph)
        intervals = self.compute_liveness_intervals(graph, sorted_nodes)

        active: list[tuple[int, int, int]] = []
        offsets: dict[str, int] = {}

        for node in sorted_nodes:
            size_val = _get_node_byte_size(node)
            if isinstance(size_val, str):
                continue

            size = int(size_val)
            birth_t, death_t = intervals.get(node.id, (0, 0))

            active = [b for b in active if b[2] >= birth_t]
            active.sort(key=lambda x: x[0])

            current_offset = 0
            for block_start, block_end, _ in active:
                aligned_offset = ((current_offset + align - 1) // align) * align
                if aligned_offset + size <= block_start:
                    current_offset = aligned_offset
                    break
                current_offset = max(current_offset, block_end)

            aligned_offset = ((current_offset + align - 1) // align) * align
            offsets[node.id] = aligned_offset
            active.append((aligned_offset, aligned_offset + size, death_t))

        return offsets

    def allocate_colored(self, graph: IRGraph, alignment: int = 256) -> dict[str, dict[str, int]]:
        """Allocate minimal linear buffers using interference graph register coloring.

        Args:
            graph (IRGraph): The computation graph.
            alignment (int): Memory alignment in bytes (default 256).

        Returns:
            dict[str, dict[str, int]]: Map of node_id to {'color': c, 'offset': off, 'size': sz}.
        """
        allocator = InterferenceGraphColoringAllocator(alignment=alignment)
        return allocator.allocate_colored_buffers(graph)

    def compute_peak_memory_reduction(self, graph: IRGraph, alignment: int = 256) -> dict[str, float]:
        """Compute peak memory reduction percentage using interference graph register coloring.

        Args:
            graph (IRGraph): Target computation graph (MLP, CNN, Transformer).
            alignment (int): Memory alignment in bytes.

        Returns:
            dict[str, float]: Peak allocation comparison metrics.
        """
        allocator = InterferenceGraphColoringAllocator(alignment=alignment)
        return allocator.compute_peak_memory_reduction(graph)

    def run(self, graph: IRGraph) -> bool:
        """Run the buffer allocation pass on the graph.

        Args:
            graph (IRGraph): Intermediate representation graph.

        Returns:
            bool: True if buffer attributes were modified, False otherwise.
        """
        sorted_nodes = DAGTopologicalSorter.sort(graph)
        if not sorted_nodes:
            return False

        graph_attrs = getattr(graph, "attributes", None)
        if graph_attrs is None:
            graph_attrs = {}
            graph.attributes = graph_attrs

        node_sizes: dict[str, int | str] = {}
        for node in sorted_nodes:
            node_sizes[node.id] = _get_node_byte_size(node)

        # Tag graph with register colored allocations for linear targets (WebGPU, WASM, LLVM/C++)
        if not graph_attrs.get("skip_coloring", False):
            try:
                coloring_allocator = InterferenceGraphColoringAllocator(alignment=256)
                # Build interference graph directly using already-extracted node_sizes
                ig = InterferenceGraph()
                birth = {node.id: i for i, node in enumerate(sorted_nodes)}
                l_use: dict[str, int] = {}
                for i, node in enumerate(sorted_nodes):
                    l_use[node.id] = i
                    for inp in node.inputs:
                        if inp in l_use:
                            l_use[inp] = max(l_use[inp], i)
                end_time = len(sorted_nodes)
                for out_id in getattr(graph, "outputs", []):
                    if out_id in l_use:
                        l_use[out_id] = end_time

                for node in sorted_nodes:
                    sz_val = node_sizes[node.id]
                    b_sz = int(sz_val) if isinstance(sz_val, int) else 1024
                    ig.add_node(node.id, b_sz, birth.get(node.id, 0), l_use.get(node.id, 0))
                ig.build_edges_from_intervals()
                colors = ig.color_registers()
                color_groups: dict[int, list[str]] = {}
                for nid, c in colors.items():
                    color_groups.setdefault(c, []).append(nid)
                color_sizes: dict[int, int] = {}
                for c, nids in color_groups.items():
                    max_sz = max(ig.sizes[nid] for nid in nids)
                    color_sizes[c] = ((max_sz + 256 - 1) // 256) * 256
                cur_off = 0
                c_offsets: dict[int, int] = {}
                for c in sorted(color_sizes.keys()):
                    c_offsets[c] = cur_off
                    cur_off += color_sizes[c]
                for nid, c in colors.items():
                    if nid in graph.nodes:
                        graph.nodes[nid].attributes["buffer_color"] = c
                        graph.nodes[nid].attributes["colored_offset"] = c_offsets[c]
            except Exception:
                pass

        last_use = _compute_liveness(graph, sorted_nodes)
        allocator = GreedyOffsetAllocator()
        modified = False

        graph.attributes = getattr(graph, "attributes", {})
        graph.attributes["dynamic_memory_schema"] = {"dynamic_offsets": []}

        for i, node in enumerate(sorted_nodes):
            size = node_sizes[node.id]

            if isinstance(size, str):
                var_name = getattr(node, "id", f"node_{i}")
                assigned_offset = allocator.allocate_dynamic(size, i, last_use.get(node.id, i), var_name)

                graph.attributes["dynamic_memory_schema"]["dynamic_offsets"].append(
                    {
                        "var_name": var_name,
                        "symbolic_math": size,
                        "node_id": node.id,
                    }
                )

                if node.attributes.get("buffer_offset_symbolic") != assigned_offset:
                    node.attributes["buffer_offset_symbolic"] = assigned_offset
                    node.attributes["buffer_size_symbolic"] = size
                    node.attributes["buffer_id"] = 0
                    modified = True
            else:
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


def buffer_allocation_pass(graph: IRGraph) -> bool:
    """In-place Buffer Allocation pass with dynamic shape support.

    Args:
        graph (IRGraph): Intermediate representation graph.

    Returns:
        bool: True if buffer attributes were modified, False otherwise.
    """
    return BufferAllocationPass().run(graph)
