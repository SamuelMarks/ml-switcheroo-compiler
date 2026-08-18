"""Pydantic models for edge generator configuration."""

from typing import Any, Optional

from pydantic import BaseModel


class WgslTemplateConfig(BaseModel):
    """Configuration for a WGSL template."""

    workgroup_size: Optional[list[int]] = None
    body: Optional[str] = None
    global_code: Optional[str] = None
    model_config = {"extra": "allow"}


class WgslTemplatesConfig(BaseModel):
    """Configuration for all WGSL templates."""

    templates: dict[str, WgslTemplateConfig]
    js_orchestration: dict[str, str] = {}

    def model_dump(self, *args: Any, **kwargs: Any) -> Any:
        """Return dict representation."""
        res = super().model_dump(*args, **kwargs)
        return res


class MemoryLimitsConfig(BaseModel):
    """Configuration for edge memory limits."""

    max_arenas: int = 16
    arena_size_bytes: int = 134217728
    reuse_policy: str = "greedy"


class WebglTemplatesConfig(BaseModel):
    """Configuration for all WebGL JS templates."""

    js_shader_compiler: str
    js_input_textures_setup: str
    js_input_texture_bind: str
    js_framebuffer_and_render: str


class MlirSpecConfig(BaseModel):
    """Configuration for MLIR Bytecode spec."""

    magic: str
    version: int
    producer: str
    sections: dict[str, int]
    default_dialects: list[str]


class StablehloSchemaConfig(BaseModel):
    """Configuration for StableHLO schema."""

    types: dict[str, str]
    operations: dict[str, str]
    bytecode: dict[str, Any]
    control_flow: dict[str, Any]
    op_mapping: dict[str, str]
