"""Pydantic models for edge generator configuration."""

from __future__ import annotations

from pydantic import BaseModel, Field


class WgslTemplateConfig(BaseModel):
    """Configuration for a WGSL template."""

    workgroup_size: list[int] | None = None
    body: str | None = None
    global_code: str | None = None
    model_config = {"extra": "allow"}


class WgslTemplatesConfig(BaseModel):
    """Configuration for all WGSL templates."""

    templates: dict[str, WgslTemplateConfig]
    js_orchestration: dict[str, str] = {}
    global_bindings: str | None = None

    def model_dump(self, *args: object, **kwargs: object) -> object:
        """Return dict representation."""
        res = super().model_dump(*args, **kwargs)
        return res


class MemoryLimitsConfig(BaseModel):
    """Configuration for edge memory limits."""

    max_arenas: int = 16
    arena_size_bytes: int = 134217728
    reuse_policy: str = "greedy"


class WebglTemplateConfig(BaseModel):
    """Configuration for a WebGL template."""

    body: str
    uniforms: list[str] = Field(default_factory=list)
    custom_setup: str = ""


class WebglTemplatesConfig(BaseModel):
    """Configuration for all WebGL JS templates."""

    templates: dict[str, WebglTemplateConfig | str]
    js_orchestration: dict[str, str]


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
    op_mapping: dict[str, str]


class WebrtcConfigIceServer(BaseModel):
    """Config for WebRTC ICE server."""

    urls: str


class WebrtcConfigSignaling(BaseModel):
    """Config for WebRTC signaling."""

    timeout_ms: int


class WebrtcConfig(BaseModel):
    """Config for WebRTC."""

    ice_servers: list[WebrtcConfigIceServer]
    signaling: WebrtcConfigSignaling


class WebrtcTopologyConfig(BaseModel):
    """Configuration for WebRTC Topology."""

    webrtc_config: WebrtcConfig
    templates: dict[str, str]


class WebrtcCollectivesSchemaDef(BaseModel):
    """Schema for WebRTC collectives."""

    chunk_size_bytes: int
    message_format: str


class WebrtcCollectivesConfig(BaseModel):
    """Configuration for WebRTC Collectives."""

    schema_def: WebrtcCollectivesSchemaDef
    handlers: dict[str, str]


class MemorySchemasDefaultConfig(BaseModel):
    """Default memory schema config."""

    growth_multiplier: float
    min_arena_size: int
    pointer_dtype: str
    byte_alignment: int


class MemorySchemasNdTensorStateConfig(BaseModel):
    """ND tensor state memory schema config."""

    shape_layout: str
    strides_layout: str
    offset_layout: str
    struct_template: str


class MemorySchemasJsOrchestrationTemplatesConfig(BaseModel):
    """JS orchestration memory schema config."""

    dynamic_resize: str
    runtime_offset_calc: str
    cond_branch_alloc: str
    cond_branch_dealloc: str


class MemorySchemasDataConfig(BaseModel):
    """Memory schemas data config."""

    default: MemorySchemasDefaultConfig
    nd_tensor_state: MemorySchemasNdTensorStateConfig
    js_orchestration_templates: MemorySchemasJsOrchestrationTemplatesConfig


class MemorySchemasConfig(BaseModel):
    """Configuration for memory schemas."""

    schemas: MemorySchemasDataConfig
