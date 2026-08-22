"""C++ Provider for Data-Driven Generation."""

from pathlib import Path
from typing import Any

import yaml

_CPP_TEMPLATES: dict[str, Any] = {}


def load_yaml(file_name: str) -> dict[str, Any]:
    """Load YAML file."""
    file_path = Path(__file__).parent / file_name
    with open(file_path) as f:
        from ml_switcheroo_compiler.backends.llvm_cpp.config_models import CppTemplatesConfig

        raw = yaml.safe_load(f)
        from typing import cast

        return cast(dict[str, Any], CppTemplatesConfig(**raw).model_dump())


def get_cpp_template(template_name: str) -> dict[str, Any]:
    """Get template."""
    global _CPP_TEMPLATES
    if not _CPP_TEMPLATES:
        _CPP_TEMPLATES = load_yaml("cpp_templates.yaml").get("templates", {})
    return _CPP_TEMPLATES.get(template_name, {})  # type: ignore
