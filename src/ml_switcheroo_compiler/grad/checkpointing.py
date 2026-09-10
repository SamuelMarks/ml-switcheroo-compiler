"""Gradient checkpointing and rematerialization utilities.

This module provides declarative rematerialization policy models loaded from YAML
and wrappers for checkpointing and recomputing graph activations during backprop.
"""

import os
import uuid
from collections.abc import Callable
from typing import Optional, Union

import yaml
from ml_switcheroo_ir import LogicalNode
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
