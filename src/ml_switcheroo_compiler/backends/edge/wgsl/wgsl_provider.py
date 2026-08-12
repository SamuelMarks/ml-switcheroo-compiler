"""WGSL Provider for Edge Runtime."""

from pathlib import Path
from typing import Any

import yaml

_WGSL_TEMPLATES: dict[str, Any] = {}


def load_yaml(file_name: str) -> dict[str, Any]:
    """Load a YAML file relative to this module.

    Args:
        file_name (str): The file name.

    Returns:
        dict[str, Any]: The loaded YAML dict.
    """
    file_path = Path(__file__).parent / file_name
    with open(file_path) as f:
        return yaml.safe_load(f)


def get_wgsl_template(template_name: str) -> dict[str, Any]:
    """Get a WGSL template.

    Args:
        template_name (str): The template name.

    Returns:
        dict[str, Any]: The template dictionary.
    """
    global _WGSL_TEMPLATES
    if not _WGSL_TEMPLATES:
        _WGSL_TEMPLATES = load_yaml("wgsl_templates.yaml").get("templates", {})
    return _WGSL_TEMPLATES.get(template_name, {})
