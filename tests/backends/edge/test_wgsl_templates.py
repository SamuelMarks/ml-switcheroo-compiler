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
        "out_width": 32,
        "out_height": 32,
        "in_width": 32,
        "in_height": 32,
        "kernel_w": 3,
        "kernel_h": 3,
        "channels": 16,
        "in_channels": 3,
        "stride_w": 1,
        "stride_h": 1,
        "filter_w": 3,
        "filter_h": 3,
        "pad_w": 1,
        "pad_h": 1,
        "block_size": 2,
        "stride": 1,
        "epsilon": 1e-5,
        "out_channels": 16,
        "window_w": 3,
        "window_h": 3,
        "in0": "in0",
        "condition_expr": "1 > 0",
        "true_body": "return;",
        "false_body": "return;",
        "max_iters": 10,
        "loop_body": "break;",
        "init_val": "0.0",
        "scan_op_expr": "acc + 1.0",
        "init_state": "0.0",
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


def test_unknown_wgsl_template():
    """Test unknown wgsl template returns empty."""
    from ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider import get_wgsl_template

    assert get_wgsl_template("nonexistent_template") == {}


def test_wgsl_provider_no_file():
    from unittest.mock import patch

    from ml_switcheroo_compiler.backends.edge.wgsl import wgsl_provider

    # Reset cache to force reload
    wgsl_provider._WGSL_TEMPLATES = {}

    with patch("os.path.exists", return_value=False):
        assert wgsl_provider.get_wgsl_template("any") == {}

    # Reset cache again to not affect other tests
    wgsl_provider._WGSL_TEMPLATES = {}
