"""Pydantic models for transpilation rules."""

from pydantic import BaseModel, Field


class FrameworkConfig(BaseModel):
    """Configuration for a specific framework."""

    target_module: str
    module_path: list[str]
    kwarg_map: dict[str, str] = Field(default_factory=dict)
    class_bases: dict[str, list[str]] = Field(default_factory=dict)
    method_map: dict[str, str] = Field(default_factory=dict)
    broadcast_method: str = "broadcast_to"


class TranspilerConfig(BaseModel):
    """Root configuration for transpiler."""

    frameworks: dict[str, FrameworkConfig]
    ast_to_ir_ops: dict[str, str] = Field(default_factory=dict)
    ir_to_ast_ops: dict[str, dict[str, list[str]]] = Field(default_factory=dict)


def load_transpiler_config(yaml_path: str) -> TranspilerConfig:
    """Load transpiler configuration from YAML file.

    Args:
        yaml_path (str): Path to the YAML configuration file.

    Returns:
        TranspilerConfig: The loaded configuration.
    """
    import yaml

    with open(yaml_path, encoding="utf-8") as f:
        data: object = yaml.safe_load(f)
    return TranspilerConfig(**data)
