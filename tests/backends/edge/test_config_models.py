from ml_switcheroo_compiler.backends.edge.config_models import WgslTemplateConfig, WgslTemplatesConfig
from ml_switcheroo_compiler.backends.edge.wasm_simd.config_models import WasmTemplateConfig, WasmTemplatesConfig


def test_wgsl_config_model():
    data = {"templates": {"test": {"workgroup_size": [1, 2, 3], "body": "test_body", "global_code": "test_global"}}}
    config = WgslTemplatesConfig(**data)
    dumped = config.model_dump()
    assert dumped["templates"]["test"]["workgroup_size"] == [1, 2, 3]

    t1 = WgslTemplateConfig()
    assert t1.workgroup_size is None


def test_wasm_config_model():
    data = {"templates": {"test": {"simd_unroll_factor": 4, "body": "test_body", "peel_loop": "test_peel", "global_code": "test_global"}}}
    config = WasmTemplatesConfig(**data)
    dumped = config.model_dump()
    assert dumped["templates"]["test"]["simd_unroll_factor"] == 4

    t1 = WasmTemplateConfig()
    assert t1.simd_unroll_factor is None
