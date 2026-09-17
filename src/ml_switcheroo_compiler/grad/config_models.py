"""Pydantic models for finite difference and declarative autodiff configs."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class FiniteDifferenceDtypeConfig(BaseModel):
    """Configuration for finite difference dtype."""

    epsilon: float


class FiniteDifferenceConfig(BaseModel):
    """Configuration for finite difference."""

    float32: FiniteDifferenceDtypeConfig
    float64: FiniteDifferenceDtypeConfig


class AutodiffRuleModel(BaseModel):
    """Declarative specification for an operation's symbolic VJP and JVP derivative rules."""

    opcode: str | None = Field(default=None, description="Operation opcode name.")
    cotangent_inputs: list[str] = Field(
        default_factory=lambda: ["$cotangent"],
        description="Cotangent input symbols.",
    )
    primal_outputs: list[str] = Field(
        default_factory=lambda: ["$output"],
        description="Primal output symbols.",
    )
    vjp: list[str] | str | None = Field(
        default=None,
        description="Symbolic VJP expression or per-input cotangent expressions.",
    )
    jvp: str | None = Field(
        default=None,
        description="Symbolic JVP tangent expression.",
    )
    description: str | None = Field(
        default=None,
        description="Documentation or mathematical justification for the rule.",
    )
    model_config = ConfigDict(extra="allow")


class HigherOrderAutodiffModel(BaseModel):
    """Declarative specification for higher-order derivatives such as HVPs."""

    opcode: str | None = Field(default=None, description="Operation opcode name.")
    hvp: str | None = Field(
        default=None,
        description="Symbolic Hessian-vector product expression.",
    )
    jvp_order: int = Field(
        default=1,
        description="Maximum supported derivative expansion order for JVP.",
    )
    vjp_order: int = Field(
        default=1,
        description="Maximum supported derivative expansion order for VJP.",
    )
    rewrite_rules: list[dict[str, str]] = Field(
        default_factory=list,
        description="Multi-order rewrite rules for symbolic graph expansion.",
    )
    model_config = ConfigDict(extra="allow")


class AutodiffRulesManifestModel(BaseModel):
    """Root declarative manifest model containing all JVP, VJP, and higher-order rules."""

    jvp_rules: dict[str, AutodiffRuleModel] = Field(
        default_factory=dict,
        description="Dictionary mapping opcodes to symbolic JVP rules.",
    )
    vjp_rules: dict[str, AutodiffRuleModel] = Field(
        default_factory=dict,
        description="Dictionary mapping opcodes to symbolic VJP rules.",
    )
    higher_order_rules: dict[str, HigherOrderAutodiffModel] = Field(
        default_factory=dict,
        description="Dictionary mapping opcodes to higher-order autodiff rules.",
    )
    model_config = ConfigDict(extra="allow")
