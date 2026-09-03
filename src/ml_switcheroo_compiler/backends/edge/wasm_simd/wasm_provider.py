"""WASM Provider for Edge Runtime."""

from pathlib import Path
from typing import Union

import yaml

_WASM_TEMPLATES: dict[str, Union[dict[str, dict[str, str]], dict[str, str], list[str]]] = {}


def load_yaml(file_name: str) -> dict[str, Union[dict[str, dict[str, str]], dict[str, str], list[str]]]:
    """Load yaml."""
    file_path: Path = Path(__file__).parent / file_name
    with open(file_path) as f:
        from ml_switcheroo_compiler.backends.edge.wasm_simd.config_models import WasmTemplatesConfig

        raw: dict[str, Union[dict[str, dict[str, str]], dict[str, str], list[str]]] = yaml.safe_load(f)
        return WasmTemplatesConfig(**raw).model_dump()


def load_yaml_dir(dir_name: str) -> dict[str, Union[dict[str, dict[str, str]], dict[str, str], list[str]]]:
    """Load all yaml files in a directory."""
    import glob

    dir_path: Path = Path(__file__).parent / dir_name
    res = {}
    if dir_path.is_dir():
        for filename in glob.glob(str(dir_path / "*.yaml")):
            with open(filename) as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict):
                    source = data.get("templates", data)
                    for k, v in source.items():
                        if isinstance(v, dict) and "body" in v:
                            res[k] = v
                        elif isinstance(v, str):
                            res[k] = {"body": v}
                        else:
                            res[k] = v
    return {"templates": res}


def get_wasm_template(template_name: str) -> dict[str, str]:
    """Get template."""
    global _WASM_TEMPLATES
    if not _WASM_TEMPLATES:
        _WASM_TEMPLATES = load_yaml_dir("wasm_templates")

    templates: Union[dict[str, dict[str, str]], dict[str, str], list[str]] = _WASM_TEMPLATES.get("templates", {})
    if isinstance(templates, dict):
        res = templates.get(template_name, {})
        if not res:
            print(f"DEBUG_PROVIDER: template_name={template_name}, type(templates)={type(templates)}, keys={list(templates.keys())[:10]}")

        if isinstance(res, dict):
            return {str(k): str(v) for k, v in res.items()}
    return {}


def get_js_orchestration_template(name: str) -> str:
    """Get js template."""
    global _WASM_TEMPLATES
    if not _WASM_TEMPLATES:
        _WASM_TEMPLATES = load_yaml_dir("wasm_templates")

    js_orch: Union[dict[str, dict[str, str]], dict[str, str], list[str]] = _WASM_TEMPLATES.get("js_orchestration", {})
    if isinstance(js_orch, dict):
        return str(js_orch.get(name, ""))
    return ""


def get_cpp_helpers() -> list[str]:
    """Get cpp helpers."""
    global _WASM_TEMPLATES
    if not _WASM_TEMPLATES:
        _WASM_TEMPLATES = load_yaml_dir("wasm_templates")

    helpers = _WASM_TEMPLATES.get("cpp_helpers", [])
    if isinstance(helpers, list):
        return [str(h) for h in helpers]
    return []
