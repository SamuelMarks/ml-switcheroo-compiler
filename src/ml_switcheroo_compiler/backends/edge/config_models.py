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


class WebglTemplateConfig(BaseModel):
    """Configuration for a WebGL template."""

    body: str


class WebglTemplatesConfig(BaseModel):
    """Configuration for all WebGL JS templates."""

    templates: dict[str, str]
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
    bytecode: dict[str, Any]
    control_flow: dict[str, Any]
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
