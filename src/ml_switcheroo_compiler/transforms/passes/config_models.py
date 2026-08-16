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


class CostModelConfig(BaseModel):
    """Configuration for cost modeling in graph scheduling."""

    memory_costs: dict[str, int]
    compute_costs: dict[str, int]
    default_memory_cost: int
    default_compute_cost: int


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
