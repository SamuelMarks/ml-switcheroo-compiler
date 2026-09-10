"""C++ Provider for Data-Driven Generation."""

from pathlib import Path
from typing import Optional, Union

import yaml

from ml_switcheroo_compiler.backends.llvm_cpp.config_models import CppOpConfig, CppTemplateConfig, CppTemplatesConfig

_CPP_CONFIG: Optional[CppTemplatesConfig] = None


def load_cpp_config() -> CppTemplatesConfig:
    """Load and parse declarative C++ templates configuration via Pydantic model.

    Returns:
        CppTemplatesConfig: Validated declarative configuration model.
    """
    global _CPP_CONFIG
    if _CPP_CONFIG is None:
        file_path: Path = Path(__file__).parent / "cpp_templates.yaml"
        with open(file_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
            _CPP_CONFIG = CppTemplatesConfig(**raw)
    return _CPP_CONFIG


def get_cpp_template(template_name: str) -> dict[str, Union[str, object]]:
    """Retrieve template dictionary by template identifier.

    Args:
        template_name (str): The name of the template.

    Returns:
        dict[str, Union[str, object]]: Template specification mapping.
    """
    cfg = load_cpp_config()
    tmpl: Optional[CppTemplateConfig] = cfg.templates.get(template_name)
    if tmpl is not None:
        return tmpl.model_dump()
    return {}


def get_cpp_operation(op_name: str) -> Optional[CppOpConfig]:
    """Retrieve declarative operation compute specification.

    Args:
        op_name (str): Operation identifier (e.g. 'sin', 'add').

    Returns:
        Optional[CppOpConfig]: Operation specification if registered.
    """
    cfg = load_cpp_config()
    return cfg.operations.get(op_name.lower())


def get_cpp_prelude() -> str:
    """Retrieve C++ prelude containing headers and data structures.

    Returns:
        str: C++ prelude source string.
    """
    cfg = load_cpp_config()
    return cfg.prelude or ""
