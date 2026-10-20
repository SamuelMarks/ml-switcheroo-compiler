"""Config models for serialization formats."""

from __future__ import annotations

from pydantic import BaseModel


class FormatSpecConfig(BaseModel):
    """Configuration for format specs."""

    dtype_map: dict[str, str]


class SerializationSchemaConfig(BaseModel):
    """Configuration for serialization schema."""

    safetensors: FormatSpecConfig
