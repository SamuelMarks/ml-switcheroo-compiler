"""C++ Provider for Data-Driven Generation."""

from pathlib import Path
from typing import Any

import yaml

_CPP_TEMPLATES: dict[str, Any] = {}


def load_yaml(file_name: str) -> dict[str, Any]:
    """Load YAML file."""
    file_path = Path(__file__).parent / file_name
    with open(file_path) as f:
        return yaml.safe_load(f)  # type: ignore


def get_cpp_template(template_name: str) -> dict[str, Any]:
    """Get template."""
    global _CPP_TEMPLATES
    if not _CPP_TEMPLATES:
        _CPP_TEMPLATES = load_yaml("cpp_templates.yaml").get("templates", {})
    return _CPP_TEMPLATES.get(template_name, {})  # type: ignore
