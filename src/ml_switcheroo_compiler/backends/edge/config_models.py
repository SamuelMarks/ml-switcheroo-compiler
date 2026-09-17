"""Pydantic models for edge generator configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class WorkgroupLayoutConfig(BaseModel):
    """Configuration for WGSL workgroup grid layout.

    Attributes:
        workgroup_size (list[int]): 3D dimensions of workgroup (x, y, z).
        max_workgroup_invocations (int): Maximum threads per workgroup.
        dynamic_dispatch_rules (dict[str, str]): Dispatch dimension expressions.
    """

    model_config = ConfigDict(extra="allow")

    workgroup_size: list[int] = Field(default_factory=lambda: [64, 1, 1])
    max_workgroup_invocations: int = 256
    dynamic_dispatch_rules: dict[str, str] = Field(default_factory=dict)


class StorageBufferBindingConfig(BaseModel):
    """Configuration for a WGSL storage buffer binding.

    Attributes:
        group (int): Bind group index.
        binding (int): Binding index within the group.
        access (str): Memory access mode ('read', 'read_write', 'write').
        dtype (str): Primitive scalar data type.
    """

    model_config = ConfigDict(extra="allow")

    group: int = 0
    binding: int = 0
    access: Literal["read", "read_write", "write"] = "read"
    dtype: Literal["f32", "i32", "u32"] = "f32"


class UniformBlockConfig(BaseModel):
    """Configuration for a WGSL uniform block.

    Attributes:
        group (int): Bind group index.
        binding (int): Binding index within the group.
        fields (dict[str, str]): Field names and WGSL types.
        byte_stride (int): Memory alignment byte stride.
    """

    model_config = ConfigDict(extra="allow")

    group: int = 0
    binding: int = 0
    fields: dict[str, str] = Field(default_factory=dict)
    byte_stride: int = 16


class MemoryAlignmentRuleConfig(BaseModel):
    """Hardware memory alignment constraints for WebGPU buffers.

    Attributes:
        struct_alignment_bytes (int): Alignment boundary for WGSL structs.
        buffer_binding_alignment_bytes (int): Hardware offset alignment for dynamic bindings.
        storage_buffer_offset_alignment (int): Storage buffer offset alignment (min 256 bytes).
    """

    model_config = ConfigDict(extra="allow")

    struct_alignment_bytes: int = 16
    buffer_binding_alignment_bytes: int = 256
    storage_buffer_offset_alignment: int = 256


class WgslTemplateConfig(BaseModel):
    """Configuration for a WGSL template.

    Attributes:
        workgroup_size (list[int] | None): 3D workgroup dimensions.
        body (str | None): Kernel body WGSL source.
        global_code (str | None): Global shader helper functions.
        workgroup_layout (WorkgroupLayoutConfig | None): Grid layout specification.
        buffer_bindings (list[StorageBufferBindingConfig]): Storage buffer declarations.
        uniform_block (UniformBlockConfig | None): Optional uniform parameters block.
        memory_alignment (MemoryAlignmentRuleConfig | None): Memory alignment requirements.
    """

    model_config = ConfigDict(extra="allow")

    workgroup_size: list[int] | None = None
    body: str | None = None
    global_code: str | None = None
    workgroup_layout: WorkgroupLayoutConfig | None = None
    buffer_bindings: list[StorageBufferBindingConfig] = Field(default_factory=list)
    uniform_block: UniformBlockConfig | None = None
    memory_alignment: MemoryAlignmentRuleConfig | None = None


class WgslTemplatesConfig(BaseModel):
    """Configuration for all WGSL templates.

    Attributes:
        templates (dict[str, WgslTemplateConfig]): Map of op name to WGSL template config.
        js_orchestration (dict[str, str]): JS orchestration code templates.
        global_bindings (str | None): Global WGSL buffer bindings string.
    """

    model_config = ConfigDict(extra="allow")

    templates: dict[str, WgslTemplateConfig]
    js_orchestration: dict[str, str] = Field(default_factory=dict)
    global_bindings: str | None = None


class WgslOpMappingConfig(BaseModel):
    """Configuration mapping an IR operation to a WGSL kernel template.

    Attributes:
        template (str): Name of the WGSL template.
        expr (str | None): Optional inline WGSL math expression.
        init_code (str | None): Optional initialization code for reductions.
        loop_code (str | None): Optional reduction loop code.
        post_loop_code (str | None): Optional post-reduction code.
        result_var (str | None): Variable name storing reduction result.
        grad_expr (str | None): Optional gradient computation expression.
    """

    model_config = ConfigDict(extra="allow")

    template: str
    expr: str | None = None
    init_code: str | None = None
    loop_code: str | None = None
    post_loop_code: str | None = None
    result_var: str | None = None
    grad_expr: str | None = None


class WgslKernelsConfig(BaseModel):
    """Declarative configuration for WebGPU WGSL kernels and bindings.

    Attributes:
        bindings (dict[str, str | list[dict[str, int | str]]]): Global and storage buffer bindings.
        memory_alignment (MemoryAlignmentRuleConfig | None): Memory alignment rules.
        op_mappings (dict[str, WgslOpMappingConfig]): Map of operation names to WGSL templates.
        templates (dict[str, WgslTemplateConfig]): Map of template names to template definitions.
    """

    model_config = ConfigDict(extra="allow")

    bindings: dict[str, str | list[dict[str, int | str]]] = Field(default_factory=dict)
    memory_alignment: MemoryAlignmentRuleConfig | None = None
    op_mappings: dict[str, WgslOpMappingConfig] = Field(default_factory=dict)
    templates: dict[str, WgslTemplateConfig] = Field(default_factory=dict)


class MemoryLimitsConfig(BaseModel):
    """Configuration for edge memory limits.

    Attributes:
        max_arenas (int): Maximum number of allocated memory arenas.
        arena_size_bytes (int): Capacity per arena in bytes.
        reuse_policy (str): Allocation reuse strategy.
    """

    max_arenas: int = 16
    arena_size_bytes: int = 134217728
    reuse_policy: str = "greedy"


class WebglTemplateConfig(BaseModel):
    """Configuration for a WebGL template.

    Attributes:
        body (str): GLSL fragment shader source.
        uniforms (list[str]): Uniform parameter names.
        custom_setup (str): Custom GL setup JavaScript.
    """

    body: str
    uniforms: list[str] = Field(default_factory=list)
    custom_setup: str = ""


class WebglTexturePackingConfig(BaseModel):
    """Configuration for WebGL float texture packing and framebuffer attachments.

    Attributes:
        internal_format (Literal["R32F", "RGBA32F"]): Internal texture format.
        format (Literal["RED", "RGBA"]): WebGL pixel format.
        type (Literal["FLOAT", "HALF_FLOAT"]): WebGL data type.
        attachment (Literal["COLOR_ATTACHMENT0", "COLOR_ATTACHMENT1"]): Framebuffer attachment point.
        viewport_width (int): Target viewport width in pixels.
        viewport_height (int): Target viewport height in pixels.
        channels (int): Number of channels (1 for R32F, 4 for RGBA32F).
    """

    model_config = ConfigDict(extra="allow")

    internal_format: Literal["R32F", "RGBA32F"] = "R32F"
    format: Literal["RED", "RGBA"] = "RED"
    type: Literal["FLOAT", "HALF_FLOAT"] = "FLOAT"
    attachment: Literal["COLOR_ATTACHMENT0", "COLOR_ATTACHMENT1"] = "COLOR_ATTACHMENT0"
    viewport_width: int = Field(default=32, ge=1, le=16384)
    viewport_height: int = Field(default=32, ge=1, le=16384)
    channels: int = Field(default=1, ge=1, le=4)

    @model_validator(mode="after")
    def validate_packing_consistency(self) -> WebglTexturePackingConfig:
        """Validate consistency between format, internal_format, and channels.

        Returns:
            WebglTexturePackingConfig: The validated instance.

        Raises:
            ValueError: If internal_format and format or channels mismatch.
        """
        if self.internal_format == "R32F" and (self.format != "RED" or self.channels != 1):
            raise ValueError("R32F internal_format requires format='RED' and channels=1")
        if self.internal_format == "RGBA32F" and (self.format != "RGBA" or self.channels != 4):
            raise ValueError("RGBA32F internal_format requires format='RGBA' and channels=4")
        return self


class WebglTemplatesConfig(BaseModel):
    """Configuration for all WebGL JS templates.

    Attributes:
        templates (dict[str, WebglTemplateConfig | str]): Map of operation names to templates.
        js_orchestration (dict[str, str]): JS orchestration code templates.
        texture_packing (WebglTexturePackingConfig): Float texture packing and framebuffer setup.
    """

    templates: dict[str, WebglTemplateConfig | str]
    js_orchestration: dict[str, str]
    texture_packing: WebglTexturePackingConfig = Field(default_factory=WebglTexturePackingConfig)


class MlirSpecConfig(BaseModel):
    """Configuration for MLIR Bytecode spec.

    Attributes:
        magic (str): 4-byte magic number.
        version (int): Bytecode format version.
        producer (str): Producer identification string.
        sections (dict[str, int]): Section IDs.
        default_dialects (list[str]): Default registered dialects.
    """

    magic: str
    version: int
    producer: str
    sections: dict[str, int]
    default_dialects: list[str]


class StablehloOpLoweringConfig(BaseModel):
    """Configuration for a specific StableHLO op lowering rule.

    Attributes:
        opcode (str): StableHLO opcode string.
        inputs (list[str]): Expected input names.
        attributes (dict[str, str]): Expected attribute names and types.
        has_reduction_region (bool): Whether the op expects a reduction body.
    """

    model_config = ConfigDict(extra="allow")

    opcode: str
    inputs: list[str] = Field(default_factory=list)
    attributes: dict[str, str] = Field(default_factory=dict)
    has_reduction_region: bool = False


class StablehloSchemaConfig(BaseModel):
    """Configuration for StableHLO schema.

    Attributes:
        types (dict[str, str]): Type mappings.
        operations (dict[str, str]): Operation specifications.
        op_mapping (dict[str, str]): IR to StableHLO op name mapping.
        lowering_rules (dict[str, StablehloOpLoweringConfig]): Detailed lowering rules.
        control_flow (dict[str, object]): Control flow operation specifications.
        bytecode (dict[str, object]): Bytecode format metadata.
    """

    model_config = ConfigDict(extra="allow")

    types: dict[str, str]
    operations: dict[str, str]
    op_mapping: dict[str, str]
    lowering_rules: dict[str, StablehloOpLoweringConfig] = Field(default_factory=dict)
    control_flow: dict[str, object] = Field(default_factory=dict)
    bytecode: dict[str, object] = Field(default_factory=dict)


class OnnxOpBuilderConfig(BaseModel):
    """Configuration for an ONNX operator builder.

    Attributes:
        op_type (str): The official ONNX operator name.
        domain (str): Domain for custom or standard ops.
        inputs (list[str]): Input argument names.
        outputs (list[str]): Output names.
        attributes (dict[str, str]): Attribute names mapped to type strings.
        min_opset (int): Minimum supported opset version.
        max_opset (int): Maximum supported opset version.
    """

    model_config = ConfigDict(extra="allow")

    op_type: str
    domain: str = ""
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    attributes: dict[str, str] = Field(default_factory=dict)
    min_opset: int = 14
    max_opset: int = 20


class OnnxSchemaConfig(BaseModel):
    """Configuration for ONNX schema definitions.

    Attributes:
        types (dict[str, int]): Primitive type name to ONNX TensorProto code mapping.
        opset_versions (dict[str, int]): Supported and default opset versions.
        operations (dict[str, OnnxOpBuilderConfig]): Operator configurations.
    """

    model_config = ConfigDict(extra="allow")

    types: dict[str, int]
    opset_versions: dict[str, int] = Field(default_factory=lambda: {"min_opset": 14, "max_opset": 20, "default_opset": 18})
    operations: dict[str, OnnxOpBuilderConfig] = Field(default_factory=dict)


class WebrtcConfigIceServer(BaseModel):
    """Config for WebRTC ICE server.

    Attributes:
        urls (str): STUN/TURN server URLs.
    """

    urls: str


class WebrtcConfigSignaling(BaseModel):
    """Config for WebRTC signaling.

    Attributes:
        timeout_ms (int): Timeout in milliseconds.
    """

    timeout_ms: int


class WebrtcConfig(BaseModel):
    """Config for WebRTC.

    Attributes:
        ice_servers (list[WebrtcConfigIceServer]): List of ICE servers.
        signaling (WebrtcConfigSignaling): Signaling configuration.
    """

    ice_servers: list[WebrtcConfigIceServer]
    signaling: WebrtcConfigSignaling


class WebrtcTopologyConfig(BaseModel):
    """Configuration for WebRTC Topology.

    Attributes:
        webrtc_config (WebrtcConfig): Connection settings.
        templates (dict[str, str]): DataChannel topology code templates.
    """

    webrtc_config: WebrtcConfig
    templates: dict[str, str]


class WebrtcCollectivesSchemaDef(BaseModel):
    """Schema for WebRTC collectives.

    Attributes:
        chunk_size_bytes (int): Max chunk size in bytes.
        message_format (str): Message serialization format.
    """

    chunk_size_bytes: int
    message_format: str


class WebrtcCollectivesConfig(BaseModel):
    """Configuration for WebRTC Collectives.

    Attributes:
        schema_def (WebrtcCollectivesSchemaDef): Wire format schema definition.
        handlers (dict[str, str]): Event handler JavaScript code.
    """

    schema_def: WebrtcCollectivesSchemaDef
    handlers: dict[str, str]


def load_webrtc_collectives(path: str | None = None) -> WebrtcCollectivesConfig:
    """Load and validate WebRTC collective configuration from YAML.

    Args:
        path (str | None): Optional filepath to webrtc_collectives.yaml.

    Returns:
        WebrtcCollectivesConfig: Validated WebRTC collectives configuration.
    """
    import os

    import yaml

    if path is None:
        path = os.path.join(os.path.dirname(__file__), "webrtc_collectives.yaml")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return WebrtcCollectivesConfig.model_validate(data)


class MemorySchemasDefaultConfig(BaseModel):
    """Default memory schema config.

    Attributes:
        growth_multiplier (float): Buffer resizing multiplier.
        min_arena_size (int): Minimum buffer arena capacity.
        pointer_dtype (str): Pointer representation data type.
        byte_alignment (int): Byte boundary alignment.
    """

    growth_multiplier: float
    min_arena_size: int
    pointer_dtype: str
    byte_alignment: int


class MemorySchemasNdTensorStateConfig(BaseModel):
    """ND tensor state memory schema config.

    Attributes:
        shape_layout (str): Shape layout descriptor.
        strides_layout (str): Strides layout descriptor.
        offset_layout (str): Offset layout descriptor.
        struct_template (str): C/WGSL struct declaration.
    """

    shape_layout: str
    strides_layout: str
    offset_layout: str
    struct_template: str


class MemorySchemasJsOrchestrationTemplatesConfig(BaseModel):
    """JS orchestration memory schema config.

    Attributes:
        dynamic_resize (str): Code for dynamic arena resizing.
        runtime_offset_calc (str): Code for calculating strides and offsets.
        cond_branch_alloc (str): Code for branch memory allocation.
        cond_branch_dealloc (str): Code for branch memory release.
    """

    dynamic_resize: str
    runtime_offset_calc: str
    cond_branch_alloc: str
    cond_branch_dealloc: str


class MemorySchemasDataConfig(BaseModel):
    """Memory schemas data config.

    Attributes:
        default (MemorySchemasDefaultConfig): Default arena settings.
        nd_tensor_state (MemorySchemasNdTensorStateConfig): Tensor state metadata.
        js_orchestration_templates (MemorySchemasJsOrchestrationTemplatesConfig): Memory JS templates.
    """

    default: MemorySchemasDefaultConfig
    nd_tensor_state: MemorySchemasNdTensorStateConfig
    js_orchestration_templates: MemorySchemasJsOrchestrationTemplatesConfig


class MemorySchemasConfig(BaseModel):
    """Configuration for memory schemas.

    Attributes:
        schemas (MemorySchemasDataConfig): Container for all memory schemas.
    """

    schemas: MemorySchemasDataConfig


class ControlFlowTemplateConfig(BaseModel):
    """Declarative specification of an edge control flow construct (loop, cond, scan).

    Attributes:
        body (str): Template body string.
        predicate_formula (Optional[str]): Parameterized formula for predicate evaluation.
        default_predicate (Optional[str]): Default predicate expression.
        default_init (Optional[str]): Default initializer expression for scans/accumulators.
        accumulation_formula (Optional[str]): Parameterized formula for scan accumulation.
    """

    body: str
    predicate_formula: str | None = None
    default_predicate: str | None = None
    default_init: str | None = None
    accumulation_formula: str | None = None


class EdgeControlFlowTemplatesConfig(BaseModel):
    """Root configuration model holding edge control flow templates.

    Attributes:
        control_flow_templates (dict[str, ControlFlowTemplateConfig]): Map of control flow op names to configs.
    """

    control_flow_templates: dict[str, ControlFlowTemplateConfig] = Field(default_factory=dict)


def load_edge_control_flow_templates(path: str) -> EdgeControlFlowTemplatesConfig:
    """Load and validate edge control flow template configurations.

    Args:
        path (str): Filepath to control_flow.yaml.

    Returns:
        EdgeControlFlowTemplatesConfig: Validated edge control flow configuration.
    """
    import yaml

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return EdgeControlFlowTemplatesConfig.model_validate(data)


class BindGroupMemoryAlignmentConfig(BaseModel):
    """Memory buffer alignment constraints for WebGPU bind groups.

    Attributes:
        min_storage_buffer_offset_alignment (int): Minimum alignment in bytes for storage buffers.
        min_uniform_buffer_offset_alignment (int): Minimum alignment in bytes for uniform buffers.
        struct_alignment_bytes (int): Alignment boundary for struct types.
        scalar_alignment_bytes (int): Alignment boundary for scalar types.
    """

    model_config = ConfigDict(extra="allow")

    min_storage_buffer_offset_alignment: int = 256
    min_uniform_buffer_offset_alignment: int = 256
    struct_alignment_bytes: int = 16
    scalar_alignment_bytes: int = 4


class BindGroupRulesConfig(BaseModel):
    """Configuration rules for dynamic WGSL bind group layout generation.

    Attributes:
        default_group (int): Default group index.
        max_storage_buffers_per_shader_stage (int): Hardware limit for storage buffers.
        max_uniform_buffers_per_shader_stage (int): Hardware limit for uniform buffers.
        dynamic_layout_generation (bool): Whether dynamic bindgroup generation is active.
        input_binding_start (int): Starting index for input bindings.
        input_access (str): WGSL storage buffer access qualifier for inputs.
        output_access (str): WGSL storage buffer access qualifier for outputs.
        uniform_access (str): WGSL uniform buffer access qualifier.
        storage_buffer_declaration (str): Template for storage buffer declaration.
        uniform_buffer_declaration (str): Template for uniform buffer declaration.
    """

    model_config = ConfigDict(extra="allow")

    default_group: int = 0
    max_storage_buffers_per_shader_stage: int = 8
    max_uniform_buffers_per_shader_stage: int = 12
    dynamic_layout_generation: bool = True
    input_binding_start: int = 0
    input_access: str = "read"
    output_access: str = "read_write"
    uniform_access: str = "read"
    storage_buffer_declaration: str = "@group({group}) @binding({binding}) var<storage, {access}> {name}: array<{dtype}>;"
    uniform_buffer_declaration: str = "@group({group}) @binding({binding}) var<uniform> {name}: {dtype};"


class BindGroupNamingConventionsConfig(BaseModel):
    """Naming conventions for dynamic WGSL buffer bindings.

    Attributes:
        input_prefix (str): Prefix for input storage buffers.
        output_prefix (str): Prefix for output storage buffers.
        uniform_prefix (str): Prefix for uniform buffers.
        single_output_name (str): Identifier for single output storage buffer.
        indexed_input_name (str): Template for indexed input storage buffer.
        indexed_output_name (str): Template for indexed output storage buffer.
    """

    model_config = ConfigDict(extra="allow")

    input_prefix: str = "buf_in"
    output_prefix: str = "buf_out"
    uniform_prefix: str = "buf_uniform"
    single_output_name: str = "buf_out_{dtype}"
    indexed_input_name: str = "buf_in{index}_{dtype}"
    indexed_output_name: str = "buf_out{index}_{dtype}"


class BindGroupSchemasConfig(BaseModel):
    """Schema model for WebGPU WGSL bind group layouts and memory alignment rules.

    Attributes:
        version (str): Schema version string.
        memory_alignment (BindGroupMemoryAlignmentConfig): Memory buffer alignment constraints.
        bindgroup_rules (BindGroupRulesConfig): Rules for dynamic binding generation.
        naming_conventions (BindGroupNamingConventionsConfig): Naming patterns.
    """

    model_config = ConfigDict(extra="allow")

    version: str = "1.0.0"
    memory_alignment: BindGroupMemoryAlignmentConfig = Field(default_factory=BindGroupMemoryAlignmentConfig)
    bindgroup_rules: BindGroupRulesConfig = Field(default_factory=BindGroupRulesConfig)
    naming_conventions: BindGroupNamingConventionsConfig = Field(default_factory=BindGroupNamingConventionsConfig)


def load_bindgroup_schemas(path: str | None = None) -> BindGroupSchemasConfig:
    """Load and validate WebGPU bind group schemas from YAML.

    Args:
        path (str, optional): Custom path to bindgroup_schemas.yaml.

    Returns:
        BindGroupSchemasConfig: Validated bindgroup schema model.
    """
    import os

    import yaml

    resolved_path = path or os.path.join(os.path.dirname(__file__), "wgsl", "bindgroup_schemas.yaml")
    with open(resolved_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return BindGroupSchemasConfig.model_validate(data)


class DtypeEmulationTypeConfig(BaseModel):
    """Configuration for emulated scalar types.

    Attributes:
        wgsl_type (str): Emulated WGSL type.
        components (dict[str, str]): Component names and primitive types.
        description (str): Human-readable explanation.
        warning (str): Warning string to emit on compile/translation.
    """

    model_config = ConfigDict(extra="allow")

    wgsl_type: str
    components: dict[str, str] = Field(default_factory=dict)
    description: str = ""
    warning: str = ""


class DtypeEmulationOpConfig(BaseModel):
    """Configuration for an emulated precision math operation.

    Attributes:
        signature (str): WGSL function signature.
        body (str): WGSL function body.
    """

    model_config = ConfigDict(extra="allow")

    signature: str
    body: str


class DtypeEmulationConfig(BaseModel):
    """Configuration schema for double precision (float64) emulation rules in WGSL.

    Attributes:
        version (str): Schema version string.
        emulation_types (dict[str, DtypeEmulationTypeConfig]): Map of type names to emulation configs.
        device_requirements (dict[str, str | bool]): WebGPU feature requirements.
        emulation_ops (dict[str, DtypeEmulationOpConfig]): Map of operation names to WGSL math implementations.
    """

    model_config = ConfigDict(extra="allow")

    version: str = "1.0.0"
    emulation_types: dict[str, DtypeEmulationTypeConfig] = Field(default_factory=dict)
    device_requirements: dict[str, str | bool] = Field(default_factory=dict)
    emulation_ops: dict[str, DtypeEmulationOpConfig] = Field(default_factory=dict)


def load_dtype_emulation_config(path: str | None = None) -> DtypeEmulationConfig:
    """Load and validate double-precision emulation rules from YAML.

    Args:
        path (str, optional): Custom path to dtype_emulation.yaml.

    Returns:
        DtypeEmulationConfig: Validated dtype emulation configuration model.
    """
    import os

    import yaml

    resolved_path = path or os.path.join(os.path.dirname(__file__), "wgsl", "dtype_emulation.yaml")
    with open(resolved_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return DtypeEmulationConfig.model_validate(data)
