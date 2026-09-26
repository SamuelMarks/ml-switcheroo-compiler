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


class CpuCostModel:
    """CPU hardware cost model calibrated against cache hierarchy and vectorization throughput."""

    def __init__(
        self,
        l1_cache_bytes: int = 32 * 1024,
        l2_cache_bytes: int = 512 * 1024,
        l3_cache_bytes: int = 16 * 1024 * 1024,
        simd_width_bytes: int = 64,
        compute_heavy_threshold: int = 250,
        heavy_interleave_penalty: int = 300,
        light_interleave_penalty: int = 50,
    ) -> None:
        """Initialize CpuCostModel with cache hierarchy parameters.

        Args:
            l1_cache_bytes (int): L1 data cache capacity in bytes.
            l2_cache_bytes (int): L2 cache capacity in bytes.
            l3_cache_bytes (int): L3 shared cache capacity in bytes.
            simd_width_bytes (int): SIMD vector register width in bytes.
            compute_heavy_threshold (int): Threshold score for heavy compute operations.
            heavy_interleave_penalty (int): Scheduling penalty for consecutive heavy ops.
            light_interleave_penalty (int): Scheduling penalty for consecutive light ops.
        """
        self.l1_cache_bytes: int = l1_cache_bytes
        self.l2_cache_bytes: int = l2_cache_bytes
        self.l3_cache_bytes: int = l3_cache_bytes
        self.simd_width_bytes: int = simd_width_bytes
        self._compute_heavy_threshold: int = compute_heavy_threshold
        self._heavy_interleave_penalty: int = heavy_interleave_penalty
        self._light_interleave_penalty: int = light_interleave_penalty

    @property
    def compute_heavy_threshold(self) -> int:
        """Threshold for heavy compute cost."""
        return self._compute_heavy_threshold

    @property
    def heavy_interleave_penalty(self) -> int:
        """Penalty for sequential heavy nodes."""
        return self._heavy_interleave_penalty

    @property
    def light_interleave_penalty(self) -> int:
        """Penalty for sequential light nodes."""
        return self._light_interleave_penalty

    def get_memory_cost(self, node: IRNode) -> int | str:
        """Calculate memory cost in bytes with CPU cache miss penalties.

        Args:
            node (IRNode): The target IR node.

        Returns:
            int | str: Estimated active memory footprint in bytes.
        """
        dtype: str = str(node.attributes.get("dtype", "float32"))
        dtype_size: int = 8 if "64" in dtype else (2 if ("16" in dtype or "bf16" in dtype) else 4)

        shape: tuple[int | str, ...] | list[int | str] | None = getattr(node, "shape_metadata", None)
        if shape is None:
            return dtype_size

        is_dynamic: bool = bool(getattr(node, "is_dynamic_shape", False))
        has_symbolic_dim: bool = any(isinstance(d, str) for d in shape)

        if not is_dynamic and not has_symbolic_dim:
            size: int = 1
            for dim in shape:
                size *= max(1, int(dim))
            byte_size: int = size * dtype_size
            if byte_size > self.l3_cache_bytes:
                return int(byte_size * 1.25)
            if byte_size > self.l2_cache_bytes:
                return int(byte_size * 1.1)
            return byte_size

        dims: list[str] = [str(d) for d in shape]
        return " * ".join(dims) + f" * {dtype_size}"

    def get_compute_cost(self, node: IRNode) -> int:
        """Calculate CPU compute cost calibrated for SIMD vector execution.

        Args:
            node (IRNode): The target IR node.

        Returns:
            int: Compute cost heuristic.
        """
        dtype: str = str(node.attributes.get("dtype", "float32"))
        dtype_size: int = 8 if "64" in dtype else 4
        vector_factor: int = max(1, self.simd_width_bytes // dtype_size)

        op: str = node.op_type
        shape = getattr(node, "shape_metadata", None)
        elems: int = 1
        if shape and not any(isinstance(d, str) for d in shape):
            for dim in shape:
                elems *= max(1, int(dim))

        if op in ("MatMul", "BatchMatMul", "Einsum", "Dot"):
            return max(500, int((elems**1.5) / vector_factor))
        if op in ("Conv2D", "Conv3D", "Conv"):
            return max(800, int(elems * 8 / vector_factor))
        if op in ("ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin"):
            return max(20, int(elems * 2 / vector_factor))
        if op in ("Add", "Sub", "Mul", "Div", "Relu", "Sigmoid", "Exp", "Log"):
            return max(5, int(elems / vector_factor))
        return 50


class GpuCostModel:
    """GPU hardware cost model calibrated against Roofline model, SM occupancy, and HBM memory bandwidth."""

    def __init__(
        self,
        hbm_bandwidth_gbps: float = 900.0,
        peak_tflops: float = 312.0,
        launch_latency_ns: int = 5000,
        sm_count: int = 108,
        compute_heavy_threshold: int = 400,
        heavy_interleave_penalty: int = 800,
        light_interleave_penalty: int = 150,
    ) -> None:
        """Initialize GpuCostModel with GPU architecture metrics.

        Args:
            hbm_bandwidth_gbps (float): High Bandwidth Memory peak bandwidth in GB/s.
            peak_tflops (float): Peak compute throughput in TFLOPs.
            launch_latency_ns (int): Kernel launch overhead latency in nanoseconds.
            sm_count (int): Number of streaming multiprocessors.
            compute_heavy_threshold (int): Threshold score for heavy compute operations.
            heavy_interleave_penalty (int): Scheduling penalty for consecutive heavy ops.
            light_interleave_penalty (int): Scheduling penalty for consecutive light ops.
        """
        self.hbm_bandwidth_gbps: float = hbm_bandwidth_gbps
        self.peak_tflops: float = peak_tflops
        self.launch_latency_ns: int = launch_latency_ns
        self.sm_count: int = sm_count
        self._compute_heavy_threshold: int = compute_heavy_threshold
        self._heavy_interleave_penalty: int = heavy_interleave_penalty
        self._light_interleave_penalty: int = light_interleave_penalty

    @property
    def compute_heavy_threshold(self) -> int:
        """Threshold for heavy compute cost."""
        return self._compute_heavy_threshold

    @property
    def heavy_interleave_penalty(self) -> int:
        """Penalty for sequential heavy nodes."""
        return self._heavy_interleave_penalty

    @property
    def light_interleave_penalty(self) -> int:
        """Penalty for sequential light nodes."""
        return self._light_interleave_penalty

    def get_memory_cost(self, node: IRNode) -> int | str:
        """Calculate GPU global device memory footprint in bytes.

        Args:
            node (IRNode): The target IR node.

        Returns:
            int | str: Estimated memory size in bytes.
        """
        dtype: str = str(node.attributes.get("dtype", "float32"))
        dtype_size: int = 8 if "64" in dtype else (2 if ("16" in dtype or "bf16" in dtype) else 4)

        shape = getattr(node, "shape_metadata", None)
        if shape is None:
            return dtype_size

        is_dynamic: bool = bool(getattr(node, "is_dynamic_shape", False))
        has_symbolic_dim: bool = any(isinstance(d, str) for d in shape)

        if not is_dynamic and not has_symbolic_dim:
            size: int = 1
            for dim in shape:
                size *= max(1, int(dim))
            return size * dtype_size

        dims: list[str] = [str(d) for d in shape]
        return " * ".join(dims) + f" * {dtype_size}"

    def get_compute_cost(self, node: IRNode) -> int:
        """Calculate GPU compute cost calibrated against Roofline throughput and launch overhead.

        Args:
            node (IRNode): The target IR node.

        Returns:
            int: Compute cost heuristic score.
        """
        launch_cost: int = int(self.launch_latency_ns / 10)
        op: str = node.op_type
        shape = getattr(node, "shape_metadata", None)
        elems: int = 1
        if shape and not any(isinstance(d, str) for d in shape):
            for dim in shape:
                elems *= max(1, int(dim))

        if op in ("MatMul", "BatchMatMul", "Einsum", "Dot"):
            return launch_cost + max(400, int((elems**1.3) / (self.sm_count * 2)))
        if op in ("Conv2D", "Conv3D", "Conv"):
            return launch_cost + max(600, int(elems * 4 / self.sm_count))
        if op in ("AllReduce", "ReduceScatter", "AllGather"):
            return launch_cost + 1000
        if op in ("Add", "Sub", "Mul", "Div", "Relu", "Sigmoid"):
            return launch_cost + max(1, int(elems / (self.hbm_bandwidth_gbps * 100)))
        return launch_cost + 50


class EdgeCostModel:
    """Edge and WebAssembly hardware cost model calibrated for constrained linear memory."""

    def __init__(
        self,
        max_linear_memory_bytes: int = 512 * 1024 * 1024,
        allocation_penalty: int = 200,
        compute_heavy_threshold: int = 150,
        heavy_interleave_penalty: int = 250,
        light_interleave_penalty: int = 40,
    ) -> None:
        """Initialize EdgeCostModel with memory limits and penalties.

        Args:
            max_linear_memory_bytes (int): Maximum addressable linear memory limit in bytes.
            allocation_penalty (int): Memory allocation penalty score.
            compute_heavy_threshold (int): Threshold for heavy operations.
            heavy_interleave_penalty (int): Penalty for consecutive heavy operations.
            light_interleave_penalty (int): Penalty for consecutive light operations.
        """
        self.max_linear_memory_bytes: int = max_linear_memory_bytes
        self.allocation_penalty: int = allocation_penalty
        self._compute_heavy_threshold: int = compute_heavy_threshold
        self._heavy_interleave_penalty: int = heavy_interleave_penalty
        self._light_interleave_penalty: int = light_interleave_penalty

    @property
    def compute_heavy_threshold(self) -> int:
        """Threshold for heavy compute cost."""
        return self._compute_heavy_threshold

    @property
    def heavy_interleave_penalty(self) -> int:
        """Penalty for sequential heavy nodes."""
        return self._heavy_interleave_penalty

    @property
    def light_interleave_penalty(self) -> int:
        """Penalty for sequential light nodes."""
        return self._light_interleave_penalty

    def get_memory_cost(self, node: IRNode) -> int | str:
        """Calculate memory cost heavily penalizing consumption exceeding constrained limits.

        Args:
            node (IRNode): The target IR node.

        Returns:
            int | str: Estimated memory cost in bytes.
        """
        dtype: str = str(node.attributes.get("dtype", "float32"))
        dtype_size: int = 8 if "64" in dtype else 4

        shape = getattr(node, "shape_metadata", None)
        if shape is None:
            return dtype_size

        is_dynamic: bool = bool(getattr(node, "is_dynamic_shape", False))
        has_symbolic_dim: bool = any(isinstance(d, str) for d in shape)

        if not is_dynamic and not has_symbolic_dim:
            size: int = 1
            for dim in shape:
                size *= max(1, int(dim))
            byte_size: int = size * dtype_size
            if byte_size > self.max_linear_memory_bytes // 2:
                return int(byte_size * 2) + self.allocation_penalty
            return byte_size + self.allocation_penalty

        dims: list[str] = [str(d) for d in shape]
        return " * ".join(dims) + f" * {dtype_size}"

    def get_compute_cost(self, node: IRNode) -> int:
        """Calculate Edge single-thread scalar compute cost.

        Args:
            node (IRNode): The target IR node.

        Returns:
            int: Compute cost score.
        """
        op: str = node.op_type
        shape = getattr(node, "shape_metadata", None)
        elems: int = 1
        if shape and not any(isinstance(d, str) for d in shape):
            for dim in shape:
                elems *= max(1, int(dim))

        if op in ("MatMul", "BatchMatMul", "Conv2D", "Conv"):
            return max(300, int(elems * 2))
        if op in ("ReduceSum", "ReduceMean"):
            return max(15, int(elems))
        if op in ("Add", "Sub", "Mul", "Div", "Relu"):
            return max(2, int(elems // 4))
        return 40


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


class GraphSchedulingPass:
    """Graph scheduling pass to minimize peak active tensor memory."""

    def __init__(self, cost_model: CostModel | None = None) -> None:
        """Initialize GraphSchedulingPass.

        Args:
            cost_model (CostModel, optional): Cost model instance.
        """
        self.cost_model: CostModel = cost_model if cost_model is not None else DefaultCostModel()

    def calculate_peak_memory(self, graph: IRGraph, schedule: list[str]) -> int:
        """Simulate execution along the given schedule and compute peak active memory in bytes.

        Args:
            graph (IRGraph): The intermediate representation graph.
            schedule (list[str]): Topological node execution order.

        Returns:
            int: Peak active tensor memory in bytes.
        """
        consumers, remaining_uses, _ = _build_adjacency_lists(graph)
        current_mem = 0
        peak_mem = 0
        for nid in schedule:
            node = graph.nodes[nid]
            current_mem += self.cost_model.get_memory_cost(node)
            peak_mem = max(peak_mem, current_mem)
            for inp in node.inputs:
                if inp in remaining_uses:
                    remaining_uses[inp] -= 1
                    if remaining_uses[inp] == 0:
                        current_mem -= self.cost_model.get_memory_cost(graph.nodes[inp])
        return peak_mem

    def schedule(self, graph: IRGraph) -> list[str]:
        """Compute memory-minimizing topological schedule.

        Args:
            graph (IRGraph): The IR graph.

        Returns:
            list[str]: Scheduled node IDs.
        """
        consumers, remaining_uses, in_degree = _build_adjacency_lists(graph)
        ready_nodes = [node_id for node_id, deg in in_degree.items() if deg == 0]
        scheduled_order: list[str] = []
        last_was_compute_heavy = False
        while ready_nodes:
            best_idx, best_node_id = _select_best_node(ready_nodes, graph, self.cost_model, remaining_uses, last_was_compute_heavy)
            ready_nodes.pop(best_idx)
            scheduled_order.append(best_node_id)
            node = graph.nodes[best_node_id]
            comp_cost = self.cost_model.get_compute_cost(node)
            last_was_compute_heavy = comp_cost > self.cost_model.compute_heavy_threshold
            _update_degrees_and_uses(node, best_node_id, consumers, remaining_uses, in_degree, ready_nodes)
        return scheduled_order

    def run(self, graph: IRGraph) -> bool:
        """Reorder graph nodes for optimal execution schedule.

        Args:
            graph (IRGraph): The input graph to mutate.

        Returns:
            bool: True if the graph was modified, False otherwise.
        """
        scheduled_order = self.schedule(graph)
        if len(scheduled_order) != len(graph.nodes):
            return False

        order_modified = scheduled_order != list(graph.nodes.keys())
        if order_modified:
            new_nodes = {node_id: graph.nodes[node_id] for node_id in scheduled_order}
            graph.nodes.clear()
            graph.nodes.update(new_nodes)

        # Populate node.stream for asynchronous stream scheduling
        stream_modified = False
        stream_counter = 0
        for nid in scheduled_order:
            node = graph.nodes[nid]
            if getattr(node, "stream", None) is None:
                parent_streams: set[str] = {str(graph.nodes[inp].stream) for inp in node.inputs if inp in graph.nodes and getattr(graph.nodes[inp], "stream", None) is not None}
                if len(parent_streams) == 1:
                    node.stream = next(iter(parent_streams))
                    stream_modified = True
                elif not parent_streams:
                    node.stream = f"stream_{stream_counter}"
                    stream_counter += 1
                    stream_modified = True
                else:
                    node.stream = min(parent_streams)
                    stream_modified = True

        return order_modified or stream_modified


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
    return GraphSchedulingPass().run(graph)
