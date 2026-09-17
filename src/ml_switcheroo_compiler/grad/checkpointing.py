"""Gradient checkpointing and rematerialization utilities.

This module provides declarative rematerialization policy models loaded from YAML
and wrappers for checkpointing and recomputing graph activations during backprop.
"""

import os
import uuid
from collections.abc import Callable
from typing import Optional, Union

import yaml
from ml_switcheroo_ir import LogicalGraph, LogicalNode
from pydantic import BaseModel, ConfigDict, Field

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops import control_flow_utils
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor


class RematerializationPolicyModel(BaseModel):
    """Declarative rematerialization policy specification."""

    description: Optional[str] = Field(default=None, description="Policy summary.")
    target_ops: list[str] = Field(default_factory=list, description="Target operations eligible for recomputation.")
    high_cost_ops: list[str] = Field(default_factory=list, description="Operations too expensive to recompute.")
    max_activation_bytes: Optional[int] = Field(default=None, description="Activation byte threshold.")
    evict_activations: bool = Field(default=True, description="Whether to evict activations from forward memory.")
    model_config = ConfigDict(extra="allow")


class RematerializationRulesConfig(BaseModel):
    """Container schema for all rematerialization policies."""

    default_policy: str = Field(default="recompute_all", description="Default policy name.")
    policies: dict[str, RematerializationPolicyModel] = Field(default_factory=dict, description="Named policies mapping.")
    target_ops: list[str] = Field(default_factory=list, description="Default target operations.")
    high_cost_ops: list[str] = Field(default_factory=list, description="Default high cost operations.")
    model_config = ConfigDict(extra="allow")


_CACHED_REMAT_RULES: Optional[RematerializationRulesConfig] = None


def load_rematerialization_rules(path: Optional[str] = None) -> RematerializationRulesConfig:
    """Load and validate declarative rematerialization rules from YAML.

    Args:
        path (Optional[str]): Optional custom file path.

    Returns:
        RematerializationRulesConfig: Validated declarative configuration.
    """
    global _CACHED_REMAT_RULES
    if _CACHED_REMAT_RULES is not None and path is None:
        return _CACHED_REMAT_RULES

    target_path = path or os.path.join(os.path.dirname(__file__), "rematerialization_rules.yaml")
    if not os.path.exists(target_path):
        target_path = os.path.join(os.path.dirname(__file__), "..", "transforms", "passes", "rematerialization_rules.yaml")

    raw_data: dict[str, object] = {}
    if os.path.exists(target_path):
        with open(target_path, encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}

    cfg = RematerializationRulesConfig.model_validate(raw_data)
    if path is None:
        _CACHED_REMAT_RULES = cfg
    return cfg


def get_rematerialization_policy(
    policy: Optional[Union[str, RematerializationPolicyModel]] = None,
) -> RematerializationPolicyModel:
    """Retrieve and validate a rematerialization policy.

    Args:
        policy (Optional[Union[str, RematerializationPolicyModel]]): Policy name or instance.

    Returns:
        RematerializationPolicyModel: The resolved policy model.

    Raises:
        ValueError: If policy name is unrecognized.
    """
    if isinstance(policy, RematerializationPolicyModel):
        return policy

    cfg = load_rematerialization_rules()
    policy_name = policy or cfg.default_policy

    if policy_name in cfg.policies:
        return cfg.policies[policy_name]

    if policy is None and cfg.target_ops:
        return RematerializationPolicyModel(
            target_ops=cfg.target_ops,
            high_cost_ops=cfg.high_cost_ops,
            evict_activations=True,
        )

    raise ValueError(f"Rematerialization policy '{policy_name}' not found in configuration.")


def checkpoint(
    fun: Callable[..., object],
    policy: Optional[Union[str, RematerializationPolicyModel]] = None,
) -> Callable[..., object]:
    """Apply gradient checkpointing / rematerialization to a function.

    Args:
        fun (Callable[..., object]): The function to checkpoint.
        policy (Optional[Union[str, RematerializationPolicyModel]]): Checkpointing policy.

    Returns:
        Callable[..., object]: The checkpointed function.
    """
    active_policy = get_rematerialization_policy(policy)

    def wrapper(*args: object, **kwargs: object) -> object:
        """Execute checkpointed function wrapper.

        Args:
            *args (object): Positional arguments.
            **kwargs (object): Keyword arguments.

        Returns:
            object: Computed result or proxy tensor.
        """
        if config.eager_mode or not global_tracing_state.is_tracing:
            return fun(*args, **kwargs)

        tensor_args = [a for a in args if isinstance(a, Tensor)]
        fwd_block = control_flow_utils._trace_function(fun, tuple(tensor_args), f"checkpoint_{uuid.uuid4().hex[:6]}")

        out_node_id = fwd_block.outputs[0]
        nodes_dict = {n.id: n for n in (fwd_block.nodes if isinstance(fwd_block.nodes, list) else fwd_block.nodes.values())}
        out_node = nodes_dict[out_node_id]
        real_out_node = nodes_dict[out_node.inputs[0]]

        shape = real_out_node.shape_metadata
        dtype = "float32"
        if hasattr(real_out_node, "attributes") and "dtype" in real_out_node.attributes:
            dtype = real_out_node.attributes["dtype"]
        elif tensor_args:
            dtype = getattr(getattr(tensor_args[0], "dtype", None), "value", "float32")

        device = "cpu"
        if tensor_args:
            device = getattr(tensor_args[0], "device", "cpu")

        out_id = str(uuid.uuid4())
        node = LogicalNode(
            id=out_id,
            op_type="Checkpoint",
            inputs=[a.data.id for a in tensor_args if hasattr(a, "data") and hasattr(a.data, "id")],
            attributes={
                "subgraph": fwd_block,
                "fun": fun,
                "evicted": active_policy.evict_activations,
                "policy": active_policy.model_dump(),
            },
            shape_metadata=shape,
        )
        global_tracing_state.add_node(node)

        proxy = ProxyTensor(id=out_id, shape=shape, dtype=dtype)
        return Tensor(proxy, TensorConfig(shape, DType(dtype), device))

    return wrapper


def remat(
    fun: Callable[..., object],
    policy: Optional[Union[str, RematerializationPolicyModel]] = None,
) -> Callable[..., object]:
    """Gradient checkpointing / rematerialization alias.

    Args:
        fun (Callable[..., object]): The function to rematerialize.
        policy (Optional[Union[str, RematerializationPolicyModel]]): Checkpointing policy.

    Returns:
        Callable[..., object]: The rematerialized function.
    """
    return checkpoint(fun, policy=policy)


def recompute_grad(
    fun: Callable[..., object],
    policy: Optional[Union[str, RematerializationPolicyModel]] = None,
) -> Callable[..., object]:
    """Gradient checkpointing / rematerialization alias.

    Args:
        fun (Callable[..., object]): The function to recompute gradients for.
        policy (Optional[Union[str, RematerializationPolicyModel]]): Checkpointing policy.

    Returns:
        Callable[..., object]: The recomputing function.
    """
    return checkpoint(fun, policy=policy)


_CACHED_COST_MODEL: Optional[dict[str, object]] = None


def load_cost_models(path: Optional[str] = None) -> dict[str, object]:
    """Load operator compute costs and memory footprints from cost_models.yaml.

    Args:
        path (Optional[str]): Custom path to cost_models.yaml.

    Returns:
        dict[str, object]: Parsed cost models dictionary.
    """
    global _CACHED_COST_MODEL
    if _CACHED_COST_MODEL is not None and path is None:
        return _CACHED_COST_MODEL

    target_path = path or os.path.join(os.path.dirname(__file__), "..", "transforms", "passes", "cost_models.yaml")
    raw_data: dict[str, object] = {}
    if os.path.exists(target_path):
        with open(target_path, encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}
    if path is None:
        _CACHED_COST_MODEL = raw_data
    return raw_data


def _evaluate_node_costs(
    node: LogicalNode,
    memory_sizes: dict[str, object],
    heavy_ops: set[str],
    light_ops: set[str],
    heavy_cost: int,
    light_cost: int,
    default_cost: int,
) -> tuple[int, int]:
    """Compute memory footprint bytes and recomputation FLOP cost for a logical node.

    Args:
        node (LogicalNode): The IR node.
        memory_sizes (dict[str, object]): Mapping from dtype string to byte size.
        heavy_ops (set[str]): Set of compute-heavy operator substrings.
        light_ops (set[str]): Set of compute-light operator substrings.
        heavy_cost (int): Base cost of heavy operators.
        light_cost (int): Base cost of light operators.
        default_cost (int): Base cost of normal operators.

    Returns:
        tuple[int, int]: Tuple of (footprint_bytes, flop_cost).
    """
    shape = getattr(node, "shape_metadata", None) or ()
    num_elements = 1
    for dim in shape:
        if isinstance(dim, int) and dim > 0:
            num_elements *= dim

    dtype_str = "float32"
    if hasattr(node, "attributes") and "dtype" in node.attributes:
        dtype_str = str(node.attributes["dtype"])
    dtype_size = int(memory_sizes.get(dtype_str, 4)) if isinstance(memory_sizes, dict) else 4
    footprint_bytes = num_elements * dtype_size

    op_name = node.op_type
    if any(h in op_name for h in heavy_ops):
        base_flop = heavy_cost
    elif any(light_op in op_name for light_op in light_ops):
        base_flop = light_cost
    else:
        base_flop = default_cost
    flop_cost = base_flop * max(1, num_elements)

    return footprint_bytes, flop_cost


def _solve_01_knapsack_dp(
    cap: int,
    weights: list[int],
    values: list[int],
) -> set[int]:
    """Solve 0/1 knapsack using dynamic programming to find retained item indices.

    Args:
        cap (int): Weight capacity.
        weights (list[int]): Item weights.
        values (list[int]): Item values.

    Returns:
        set[int]: Indices of retained items.
    """
    num_items = len(weights)
    max_cells = 2000
    scale = (cap + max_cells - 1) // max_cells if cap > max_cells else 1
    scaled_cap = cap // scale
    scaled_weights = [max(1, w // scale) for w in weights]

    dp = [0] * (scaled_cap + 1)
    keep = [[False] * (scaled_cap + 1) for _ in range(num_items)]

    for i in range(num_items):
        w = scaled_weights[i]
        v = values[i]
        for c in range(scaled_cap, w - 1, -1):
            if dp[c - w] + v > dp[c]:
                dp[c] = dp[c - w] + v
                keep[i][c] = True

    curr_c = scaled_cap
    retained_indices: set[int] = set()
    for i in range(num_items - 1, -1, -1):
        if keep[i][curr_c]:
            retained_indices.add(i)
            curr_c -= scaled_weights[i]
    return retained_indices


def knapsack_memory_scheduler(
    graph: LogicalGraph,
    memory_budget_bytes: int,
    cost_model: Optional[dict[str, object]] = None,
) -> list[str]:
    """Solve dynamic 0/1 knapsack problem to find optimal checkpoint cut-points minimizing recomputation FLOPs.

    Given a peak activation memory budget M_bytes, nodes retained in forward memory
    are chosen to maximize saved recomputation FLOPs subject to memory footprint <= M_bytes.
    All nodes not chosen to be retained in forward memory are marked for rematerialization.

    Args:
        graph (LogicalGraph): Computation graph to analyze and transform.
        memory_budget_bytes (int): Maximum peak activation memory in bytes.
        cost_model (Optional[dict[str, object]]): Preloaded cost model dictionary.

    Returns:
        list[str]: Node IDs designated for checkpointing/rematerialization.
    """
    cm = cost_model or load_cost_models()
    memory_sizes = cm.get("memory_sizes", {}) if isinstance(cm, dict) else {}
    compute_costs = cm.get("compute_costs", {}) if isinstance(cm, dict) else {}

    heavy_ops = set(compute_costs.get("heavy_ops", ["Conv", "MatMul"])) if isinstance(compute_costs, dict) else {"Conv", "MatMul"}
    light_ops = set(compute_costs.get("light_ops", ["Add", "Sub"])) if isinstance(compute_costs, dict) else {"Add", "Sub"}
    heavy_cost = int(compute_costs.get("heavy_cost", 1000)) if isinstance(compute_costs, dict) else 1000
    light_cost = int(compute_costs.get("light_cost", 10)) if isinstance(compute_costs, dict) else 10
    default_cost = int(compute_costs.get("default_cost", 50)) if isinstance(compute_costs, dict) else 50

    cfg = load_rematerialization_rules()
    target_ops = set(cfg.target_ops)
    high_cost_ops = set(cfg.high_cost_ops)

    candidates: list[str] = []
    weights: list[int] = []
    values: list[int] = []

    fixed_memory = 0
    total_activation_bytes = 0

    for nid, node in graph.nodes.items():
        if node.op_type in ("Input", "Constant", "Checkpoint"):
            continue

        footprint, flop_cost = _evaluate_node_costs(node, memory_sizes, heavy_ops, light_ops, heavy_cost, light_cost, default_cost)
        total_activation_bytes += footprint

        if node.op_type in high_cost_ops or (target_ops and node.op_type not in target_ops):
            fixed_memory += footprint
            continue

        candidates.append(nid)
        weights.append(footprint)
        values.append(flop_cost)

    if total_activation_bytes <= memory_budget_bytes or not candidates:
        return []

    cap = max(0, memory_budget_bytes - fixed_memory)
    retained_indices = _solve_01_knapsack_dp(cap, weights, values)

    checkpointed_nodes: list[str] = []
    for i, nid in enumerate(candidates):
        if i not in retained_indices:
            node = graph.nodes[nid]
            node.attributes["checkpoint"] = True
            node.attributes["rematerialize"] = True
            checkpointed_nodes.append(nid)

    return checkpointed_nodes


def solve_memory_budget_and_insert_checkpoints(
    graph: LogicalGraph,
    memory_budget_bytes: int,
) -> list[str]:
    """Identify peak memory activation nodes and mark them for checkpointing within a memory budget.

    Args:
        graph (LogicalGraph): Computation graph to analyze and transform.
        memory_budget_bytes (int): Maximum allowable peak activation memory in bytes.

    Returns:
        list[str]: Node IDs designated for checkpointing/rematerialization.
    """
    return knapsack_memory_scheduler(graph, memory_budget_bytes)
