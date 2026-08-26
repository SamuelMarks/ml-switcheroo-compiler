"""WGSL template provider."""

import os

import yaml

_WGSL_TEMPLATES = {}


def _load_templates() -> None:
    """Load wgsl templates."""
    global _WGSL_TEMPLATES
    if not _WGSL_TEMPLATES:
        path: str = os.path.join(os.path.dirname(__file__), "wgsl_templates.yaml")
        if os.path.exists(path):
            with open(path) as f:
                from ml_switcheroo_compiler.backends.edge.config_models import WgslTemplatesConfig

                raw_data: dict[str, str] = yaml.safe_load(f)
                _WGSL_TEMPLATES = WgslTemplatesConfig(**raw_data).model_dump()


def get_wgsl_template(name: str):
    """Get wgsl template."""
    _load_templates()
    return _WGSL_TEMPLATES.get("templates", {}).get(name, {})


def get_js_orchestration_template(name: str) -> str:
    """Get js orchestration template."""
    _load_templates()
    return str(_WGSL_TEMPLATES.get("js_orchestration", {}).get(name, ""))
