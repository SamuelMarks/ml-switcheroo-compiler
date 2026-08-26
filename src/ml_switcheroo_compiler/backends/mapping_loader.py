"""Backend YAML Mapping Loader."""

import os
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field


class KwargTranslation(BaseModel):
    """Kwarg translation."""

    target_name: str
    default_value: Any = None


class OpMappingSchema(BaseModel):
    """Op schema."""

    target_api: str
    is_method: bool = False
    kwarg_translations: dict[str, str] = Field(default_factory=dict)
    supported_dtypes: Optional[list[str]] = None
    ast_template: Optional[str] = None
    custom_code: Optional[str] = None


class BackendMappingSchema(BaseModel):
    """Backend schema."""

    backend_name: str
    operations: dict[str, OpMappingSchema]
    helpers: Optional[list[str]] = None


_MAPPING_CACHE: dict[str, BackendMappingSchema] = {}


def load_backend_mappings(backend_name: str) -> BackendMappingSchema:
    """Load mappings."""
    global _MAPPING_CACHE
    if backend_name in _MAPPING_CACHE:
        return _MAPPING_CACHE[backend_name]

    base_dir: str = os.path.dirname(os.path.abspath(__file__))
    yaml_path: str = os.path.join(base_dir, backend_name, "mappings.yaml")

    if not os.path.exists(yaml_path):
        schema: BackendMappingSchema = BackendMappingSchema(backend_name=backend_name, operations={})
        _MAPPING_CACHE[backend_name] = schema
        return schema

    with open(yaml_path) as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}

    schema = BackendMappingSchema(**data)
    _MAPPING_CACHE[backend_name] = schema
    return schema


def resolve_target_api(api_str: str, custom_code: Optional[str] = None, backend_module: Any = None) -> Any:
    """Resolve target api."""
    if api_str == "custom_op" and custom_code:
        local_env: dict[str, Any] = {"backend_module": backend_module}
        if backend_module:
            for k in dir(backend_module):
                if not k.startswith("__"):
                    local_env[k] = getattr(backend_module, k)
        try:
            return eval(custom_code, local_env)
        except Exception:
            return None

    if not api_str:
        return None

    parts: list[str] = api_str.split(".")
    try:
        import importlib

        mod: Any = importlib.import_module(parts[0])
        for p in parts[1:]:
            mod = getattr(mod, p)
        return mod
    except Exception:
        if backend_module and hasattr(backend_module, api_str):
            return getattr(backend_module, api_str)
        return None
