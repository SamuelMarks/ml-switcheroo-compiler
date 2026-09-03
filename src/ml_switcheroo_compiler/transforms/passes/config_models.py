"""Pydantic models for configuration files."""

from typing import Optional

from pydantic import BaseModel


class NodePatternConfig(BaseModel):
    """Configuration for an IR node pattern."""

    op_type: Optional[str] = None
    capture: Optional[str] = None
    inputs: Optional[list["NodePatternConfig"]] = None


class ReplacementConfig(BaseModel):
    """Configuration for replacement of matched pattern."""

    op_type: str
    inputs: list[str]
    capture_to_replace: str


class FusionPatternConfig(BaseModel):
    """Configuration for a fusion pattern."""

    pattern: NodePatternConfig
    replacement: ReplacementConfig


class ComputeCosts(BaseModel):
    """Compute costs configuration."""

    heavy_ops: list[str]
    light_ops: list[str]
    heavy_cost: int
    light_cost: int
    default_cost: int


class CostModelConfig(BaseModel):
    """Configuration for cost modeling in graph scheduling."""

    memory_sizes: dict[str, int]
    compute_costs: ComputeCosts
    compute_heavy_threshold: int
    heavy_interleave_penalty: int
    light_interleave_penalty: int


class PassConfig(BaseModel):
    """Configuration for the pass manager."""

    execution_order: list[str]
    cost_model: CostModelConfig
    fusion_patterns: dict[str, FusionPatternConfig]


class RematerializationThresholds(BaseModel):
    """Thresholds for rematerialization."""

    min_memory_bytes: int
    max_compute_to_memory_ratio: float


class RematerializationRulesConfig(BaseModel):
    """Configuration for rematerialization rules."""

    target_ops: list[str]
    high_cost_ops: list[str]
    thresholds: RematerializationThresholds


class OptimizationHeuristicsConfig(BaseModel):
    """Configuration for optimization heuristics."""

    in_place_safe_ops: list[str]


class BehaviorDescriptorsConfig(BaseModel):
    """Configuration for behavior descriptors."""

    side_effect_ops: list[str]
