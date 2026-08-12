from pathlib import Path

import yaml


def test_wgsl_templates_validity():
    template_path = Path("src/ml_switcheroo_compiler/backends/edge/wgsl/wgsl_templates.yaml")
    assert template_path.exists(), "wgsl_templates.yaml not found"

    with open(template_path) as f:
        data = yaml.safe_load(f)

    templates = data.get("templates", {})
    assert isinstance(templates, dict)

    # Try formatting each template to ensure there are no unbound template vars
    # We will provide a dummy context containing all possible expected keys
    dummy_context = {
        "nelem": 100,
        "out_offset_code": "let out_offset = 0u;",
        "in0_offset_code": "let in0_offset = 0u;",
        "in1_offset_code": "let in1_offset = 0u;",
        "in2_offset_code": "let in2_offset = 0u;",
        "expr": "1.0",
        "clean_id": "test_id",
        "TILE_SIZE": 16,
        "K": 32,
        "N": 32,
        "M": 32,
        "init_code": "var res = 0.0;",
        "start_idx": "0",
        "loop_code": "res += 1.0;",
        "post_loop_code": "res /= 2.0;",
        "result_var": "res",
        "nelem_in": 10,
    }

    for name, tpl in templates.items():
        if "body" in tpl:
            try:
                formatted = tpl["body"].format(**dummy_context)
                assert isinstance(formatted, str)
            except KeyError as e:
                raise AssertionError(f"Template '{name}' body is missing format key: {e}") from e

        if "global_code" in tpl:
            try:
                formatted = tpl["global_code"].format(**dummy_context)
                assert isinstance(formatted, str)
            except KeyError as e:
                raise AssertionError(f"Template '{name}' global_code is missing format key: {e}") from e
