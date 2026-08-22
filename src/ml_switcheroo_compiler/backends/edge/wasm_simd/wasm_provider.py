"""WASM Provider for Edge Runtime."""

from pathlib import Path
from typing import Any

import yaml

_WASM_TEMPLATES: dict[str, Any] = {}


def load_yaml(file_name: str) -> Any:
    """Load yaml."""
    file_path = Path(__file__).parent / file_name
    with open(file_path) as f:
        from ml_switcheroo_compiler.backends.edge.wasm_simd.config_models import WasmTemplatesConfig

        raw = yaml.safe_load(f)
        return WasmTemplatesConfig(**raw).model_dump()


def get_wasm_template(template_name: str) -> Any:
    """Get template."""
    global _WASM_TEMPLATES
    if not _WASM_TEMPLATES:
        _WASM_TEMPLATES = load_yaml("wasm_templates.yaml")
    return _WASM_TEMPLATES.get("templates", {}).get(template_name, {})


def get_js_orchestration_template(name: str) -> str:
    """Get js template."""
    global _WASM_TEMPLATES
    if not _WASM_TEMPLATES:
        _WASM_TEMPLATES = load_yaml("wasm_templates.yaml")
    return str(_WASM_TEMPLATES.get("js_orchestration", {}).get(name, ""))


def get_cpp_helpers() -> list[str]:
    """Get cpp helpers."""
    global _WASM_TEMPLATES
    if not _WASM_TEMPLATES:
        _WASM_TEMPLATES = load_yaml("wasm_templates.yaml")
    from typing import cast

    return cast(list[str], _WASM_TEMPLATES.get("cpp_helpers", []))
