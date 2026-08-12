import os

import yaml


def test_ops_registry_yaml_loads():
    yaml_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "src", "ml_switcheroo_compiler", "ops", "ops_registry.yaml")
    assert os.path.exists(yaml_path), "YAML registry file does not exist"
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    assert isinstance(data, dict)
    assert len(data) > 4000

    # Check that a standard op is well-formed
    assert "Add" in data
    assert "variants" in data["Add"]
    assert "llvm_cpp" in data["Add"]["variants"]


def test_pass_config_yaml_loads():
    yaml_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "src", "ml_switcheroo_compiler", "transforms", "pass_config.yaml")
    assert os.path.exists(yaml_path), "Pass config YAML does not exist"
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    assert "fusion_patterns" in data
    assert "MHAFusion" in data["fusion_patterns"]
    assert "execution_order" in data
    assert "cost_model" in data


def test_wgsl_templates_yaml_loads():
    yaml_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "src", "ml_switcheroo_compiler", "backends", "edge", "wgsl", "wgsl_templates.yaml")
    assert os.path.exists(yaml_path), "WGSL templates YAML does not exist"
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    assert "templates" in data
    assert "unary" in data["templates"]
