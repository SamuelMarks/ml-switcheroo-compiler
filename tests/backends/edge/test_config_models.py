from ml_switcheroo_compiler.backends.edge.config_models import (
    MemoryAlignmentRuleConfig,
    StorageBufferBindingConfig,
    UniformBlockConfig,
    WgslKernelsConfig,
    WgslOpMappingConfig,
    WgslTemplateConfig,
    WgslTemplatesConfig,
    WorkgroupLayoutConfig,
)
from ml_switcheroo_compiler.backends.edge.wasm_simd.config_models import (
    LaneArithmeticConfig,
    MemoryAlignmentConfig,
    WasmIntrinsicConfig,
    WasmIntrinsicsConfig,
    WasmTemplateConfig,
    WasmTemplatesConfig,
)


def test_wgsl_config_model():
    wg_layout = WorkgroupLayoutConfig(workgroup_size=[64, 1, 1], max_workgroup_invocations=256)
    buf_bind = StorageBufferBindingConfig(group=0, binding=0, access="read", dtype="f32")
    uni_block = UniformBlockConfig(group=0, binding=2, fields={"batch_size": "u32"})
    mem_align = MemoryAlignmentRuleConfig(struct_alignment_bytes=16, buffer_binding_alignment_bytes=256)

    data = {
        "templates": {
            "test": {
                "workgroup_size": [1, 2, 3],
                "body": "test_body",
                "global_code": "test_global",
                "workgroup_layout": wg_layout.model_dump(),
                "buffer_bindings": [buf_bind.model_dump()],
                "uniform_block": uni_block.model_dump(),
                "memory_alignment": mem_align.model_dump(),
            }
        }
    }
    config = WgslTemplatesConfig(**data)
    dumped = config.model_dump()
    assert dumped["templates"]["test"]["workgroup_size"] == [1, 2, 3]
    assert dumped["templates"]["test"]["workgroup_layout"]["max_workgroup_invocations"] == 256
    assert dumped["templates"]["test"]["buffer_bindings"][0]["access"] == "read"

    t1 = WgslTemplateConfig()
    assert t1.workgroup_size is None

    op_map = WgslOpMappingConfig(template="binary", expr="a + b")
    assert op_map.template == "binary"
    assert op_map.expr == "a + b"

    kernels_cfg = WgslKernelsConfig(
        bindings={"global_bindings": "// test"},
        op_mappings={"add": op_map},
        templates={"binary": WgslTemplateConfig(workgroup_size=[64, 1, 1], body="// test")},
    )
    dumped_kernels = kernels_cfg.model_dump()
    assert dumped_kernels["op_mappings"]["add"]["template"] == "binary"
    assert dumped_kernels["templates"]["binary"]["workgroup_size"] == [64, 1, 1]


def test_wasm_config_model():
    data = {
        "templates": {
            "test": {
                "simd_unroll_factor": 4,
                "body": "test_body",
                "peel_loop": "test_peel",
                "global_code": "test_global",
                "alignment": {"alignment_bytes": 16, "requires_aligned_load": True, "peeling_loop_required": True},
                "lane_arithmetic": {"lanes": 4, "element_type": "f32", "simd_intrinsic": "wasm_f32x4_add"},
            }
        }
    }
    config = WasmTemplatesConfig(**data)
    dumped = config.model_dump()
    assert dumped["templates"]["test"]["simd_unroll_factor"] == 4
    assert dumped["templates"]["test"]["alignment"]["alignment_bytes"] == 16

    t1 = WasmTemplateConfig()
    assert t1.simd_unroll_factor is None

    mem_cfg = MemoryAlignmentConfig()
    assert mem_cfg.alignment_bytes == 16
    assert mem_cfg.requires_aligned_load is True

    lane_cfg = LaneArithmeticConfig(simd_intrinsic="wasm_f32x4_mul")
    assert lane_cfg.lanes == 4
    assert lane_cfg.element_type == "f32"

    intr_cfg = WasmIntrinsicConfig(macro_name="TEST_MACRO", simd_expr="return x;", alignment=mem_cfg, lane_arithmetic=lane_cfg)
    assert intr_cfg.macro_name == "TEST_MACRO"

    intrs_cfg = WasmIntrinsicsConfig(intrinsics={"test": intr_cfg}, scalars={"abs": "return std::abs(a);"})
    assert "test" in intrs_cfg.intrinsics
    assert intrs_cfg.scalars["abs"] == "return std::abs(a);"


def test_onnx_and_stablehlo_config_models():
    from ml_switcheroo_compiler.backends.edge.config_models import (
        OnnxOpBuilderConfig,
        OnnxSchemaConfig,
        StablehloOpLoweringConfig,
        StablehloSchemaConfig,
    )

    op_builder = OnnxOpBuilderConfig(
        op_type="Relu",
        inputs=["X"],
        outputs=["Y"],
        attributes={},
        min_opset=14,
        max_opset=20,
    )
    assert op_builder.op_type == "Relu"
    assert op_builder.min_opset == 14

    onnx_schema = OnnxSchemaConfig(
        types={"float32": 1},
        opset_versions={"min_opset": 14, "max_opset": 20, "default_opset": 18},
        operations={"Relu": op_builder},
    )
    assert onnx_schema.types["float32"] == 1
    assert "Relu" in onnx_schema.operations

    sh_lowering = StablehloOpLoweringConfig(
        opcode="stablehlo.dot_general",
        inputs=["lhs", "rhs"],
        attributes={"dot_dimension_numbers": "#stablehlo.dot"},
        has_reduction_region=False,
    )
    assert sh_lowering.opcode == "stablehlo.dot_general"
    assert sh_lowering.has_reduction_region is False

    sh_schema = StablehloSchemaConfig(
        types={"float32": "f32"},
        operations={"fallback": "stablehlo.custom_call"},
        op_mapping={"Add": "stablehlo.add"},
        lowering_rules={"dot_general": sh_lowering},
    )
    assert sh_schema.types["float32"] == "f32"
    assert sh_schema.op_mapping["Add"] == "stablehlo.add"
    assert "dot_general" in sh_schema.lowering_rules


def test_edge_webrtc_collectives_config_explicit_path() -> None:
    """Test load_webrtc_collectives with an explicit path argument."""
    import os

    import ml_switcheroo_compiler.backends.edge.config_models as cm
    from ml_switcheroo_compiler.backends.edge.config_models import load_webrtc_collectives

    real_path = os.path.join(os.path.dirname(cm.__file__), "webrtc_collectives.yaml")
    cfg = load_webrtc_collectives(path=real_path)
    assert cfg is not None


def test_bindgroup_schemas_config() -> None:
    """Test loading and validating bindgroup_schemas.yaml."""
    from ml_switcheroo_compiler.backends.edge.config_models import BindGroupSchemasConfig, load_bindgroup_schemas
    from ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider import generate_dynamic_bindgroup_declarations, get_bindgroup_schemas

    cfg = load_bindgroup_schemas()
    assert isinstance(cfg, BindGroupSchemasConfig)
    assert cfg.memory_alignment.min_storage_buffer_offset_alignment == 256
    assert cfg.bindgroup_rules.max_storage_buffers_per_shader_stage == 8
    assert cfg.naming_conventions.single_output_name == "buf_out_{dtype}"

    dumped = get_bindgroup_schemas()
    assert "memory_alignment" in dumped

    # Dynamic declarations test
    decl = generate_dynamic_bindgroup_declarations(num_inputs=5, num_outputs=2)
    assert "@binding(0) var<storage, read> buf_in0_f32" in decl
    assert "@binding(3) var<storage, read_write> buf_out_f32" in decl
    assert "@binding(4) var<storage, read> buf_in3_f32" in decl
    assert "@binding(5) var<storage, read> buf_in4_f32" in decl
    assert "@binding(6) var<storage, read_write> buf_out1_f32" in decl

    decl_unary = generate_dynamic_bindgroup_declarations(num_inputs=1, num_outputs=1)
    assert "@binding(0) var<storage, read> buf_in0_f32" in decl_unary
    assert "@binding(1) var<storage, read> buf_in1_f32" in decl_unary
    assert "@binding(3) var<storage, read_write> buf_out_f32" in decl_unary


def test_dtype_emulation_config() -> None:
    """Test loading and validating dtype_emulation.yaml."""
    from ml_switcheroo_compiler.backends.edge.config_models import DtypeEmulationConfig, load_dtype_emulation_config
    from ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider import get_dtype_emulation_config

    cfg = load_dtype_emulation_config()
    assert isinstance(cfg, DtypeEmulationConfig)
    assert "float64" in cfg.emulation_types
    assert cfg.emulation_types["float64"].wgsl_type == "vec2<f32>"
    assert "two_sum" in cfg.emulation_ops
    assert "df64_add" in cfg.emulation_ops

    dumped = get_dtype_emulation_config()
    assert "emulation_types" in dumped
