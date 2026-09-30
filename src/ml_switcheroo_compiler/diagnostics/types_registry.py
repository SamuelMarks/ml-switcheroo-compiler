"""Diagnostics and types registry schema and validation for non-math metadata classes."""

from __future__ import annotations

import os
from typing import Union

import yaml
from pydantic import BaseModel

MetadataValue = Union[str, int, float, bool, list[str], list[dict[str, Union[str, bool]]]]


class TypeDefinition(BaseModel):
    """Metadata specification for a non-math class, type, or exception.

    Attributes:
        name: Name of the non-math type or class.
        description: Description of the type or exception.
        category: Classification category (exception, class, state, spec).
        attributes: Additional metadata attributes.
    """

    name: str
    description: str = ""
    category: str = "type"
    attributes: dict[str, MetadataValue] = {}


class TypesRegistryConfig(BaseModel):
    """Registry holding isolated non-math metadata types and exceptions.

    Attributes:
        types: Dictionary mapping type name to TypeDefinition.
    """

    types: dict[str, TypeDefinition] = {}


def load_types_registry(yaml_path: str | None = None) -> TypesRegistryConfig:
    """Load non-math types registry from YAML configuration.

    Args:
        yaml_path: Optional path to the types_registry.yaml file.

    Returns:
        TypesRegistryConfig: Populated types registry configuration.
    """
    if yaml_path is None:
        yaml_path = os.path.join(os.path.dirname(__file__), "types_registry.yaml")
    if not os.path.exists(yaml_path):
        return TypesRegistryConfig(types={})
    with open(yaml_path, encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
        data: dict[str, dict[str, TypeDefinition]] = loaded if isinstance(loaded, dict) else {}
    return TypesRegistryConfig(**data)


_TYPES_CACHE: set[str] | None = None


def is_non_math_type(name: str) -> bool:
    """Check whether a given operation symbol is a non-math type, class, or exception.

    Args:
        name: The name or symbol to check.

    Returns:
        bool: True if the symbol is a non-math type/exception, False otherwise.
    """
    global _TYPES_CACHE
    if _TYPES_CACHE is None:
        cfg = load_types_registry()
        _TYPES_CACHE = set(cfg.types.keys())
    return name in _TYPES_CACHE
