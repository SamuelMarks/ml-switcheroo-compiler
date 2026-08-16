"""Tests for transpiler config models."""

from pathlib import Path

from ml_switcheroo_compiler.backends.transpiler_config_models import FrameworkConfig, TranspilerConfig, load_transpiler_config


def test_framework_config() -> None:
    """Test FrameworkConfig."""
    config = FrameworkConfig(target_module="jax", module_path=["jax", "numpy"])
    assert config.target_module == "jax"
    assert config.module_path == ["jax", "numpy"]
    assert config.kwarg_map == {}
    assert config.class_bases == {}
    assert config.method_map == {}
    assert config.broadcast_method == "broadcast_to"


def test_transpiler_config() -> None:
    """Test TranspilerConfig."""
    fconfig = FrameworkConfig(target_module="jax", module_path=["jax", "numpy"])
    tconfig = TranspilerConfig(frameworks={"jax": fconfig})
    assert tconfig.frameworks["jax"].target_module == "jax"


def test_load_transpiler_config(tmp_path: Path) -> None:
    """Test loading transpiler config."""
    yaml_content = """
frameworks:
  jax:
    target_module: "jax"
    module_path: ["jax", "numpy"]
"""
    file_path = tmp_path / "config.yaml"
    with open(file_path, "w") as f:
        f.write(yaml_content)

    config = load_transpiler_config(str(file_path))
    assert "jax" in config.frameworks
    assert config.frameworks["jax"].target_module == "jax"
