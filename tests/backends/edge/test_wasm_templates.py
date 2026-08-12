from pathlib import Path

import yaml


def test_wasm_templates_validity():
    template_path = Path("src/ml_switcheroo_compiler/backends/edge/wasm_simd/wasm_templates.yaml")
    assert template_path.exists(), "wasm_templates.yaml not found"

    with open(template_path) as f:
        data = yaml.safe_load(f)

    templates = data.get("templates", {})
    assert isinstance(templates, dict)

    dummy_context = {
        "nelem": 100,
        "clean_id": "test_id",
        "op_type": "Add",
        "in0": "in0_id",
        "in1": "in1_id",
        "K": 32,
        "N": 32,
        "M": 32,
        "nelem_in": 10,
        "simd_expr": "wasm_f32x4_add(in0_val, in1_val)",
        "scalar_expr": "in0_val + in1_val",
        "init_val": "0.0",
        "final_combine": "scalar_sum_test_id",
        "math_op": "+ 1.0f",
    }

    for name, tpl in templates.items():
        if "body" in tpl:
            try:
                formatted = tpl["body"].format(**dummy_context)
                assert isinstance(formatted, str)
            except KeyError as e:
                raise AssertionError(f"Template '{name}' body is missing format key: {e}") from e
