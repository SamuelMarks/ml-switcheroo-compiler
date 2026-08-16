"""WASM Provider for Edge Runtime."""

from pathlib import Path
from typing import Any

import yaml

_WASM_TEMPLATES: dict[str, Any] = {}


def load_yaml(file_name: str) -> Any:
    """Load a YAML file relative to this module.

    Args:
        file_name (str): The file name.

    Returns:
        dict[str, Any]: The loaded YAML dict.
    """
    file_path = Path(__file__).parent / file_name
    with open(file_path) as f:
        from ml_switcheroo_compiler.backends.edge.wasm_simd.config_models import WasmTemplatesConfig

        raw = yaml.safe_load(f)
        return WasmTemplatesConfig(**raw).model_dump()


def get_wasm_template(template_name: str) -> Any:
    """Get a WASM template.

    Args:
        template_name (str): The template name.

    Returns:
        dict[str, Any]: The template dictionary.
    """
    global _WASM_TEMPLATES
    if not _WASM_TEMPLATES:
        _WASM_TEMPLATES = load_yaml("wasm_templates.yaml")
    return _WASM_TEMPLATES.get(template_name, {})
