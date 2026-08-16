"""WGSL template provider."""

import os
from typing import Any

import yaml

_WGSL_TEMPLATES: dict[str, Any] = {}


def _load_templates() -> None:
    """Load wgsl templates."""
    global _WGSL_TEMPLATES
    if not _WGSL_TEMPLATES:
        path = os.path.join(os.path.dirname(__file__), "wgsl_templates.yaml")
        if os.path.exists(path):
            with open(path) as f:
                from ml_switcheroo_compiler.backends.edge.config_models import WgslTemplatesConfig

                raw_data = yaml.safe_load(f)
                _WGSL_TEMPLATES = WgslTemplatesConfig(**raw_data).model_dump()


def get_wgsl_template(name: str) -> Any:
    """Get wgsl template."""
    _load_templates()
    if name not in _WGSL_TEMPLATES:
        return {}
    return _WGSL_TEMPLATES[name]
