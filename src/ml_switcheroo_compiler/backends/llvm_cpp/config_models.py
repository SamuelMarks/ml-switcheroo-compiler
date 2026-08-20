"""Pydantic models for C++ generator configuration."""

from typing import Any, Optional

from pydantic import BaseModel


class CppTemplateConfig(BaseModel):
    """Configuration for a C++ template."""

    body: Optional[str] = None
    includes: Optional[list[str]] = None
    model_config = {"extra": "allow"}


class CppTemplatesConfig(BaseModel):
    """Configuration for all C++ templates."""

    templates: dict[str, CppTemplateConfig]

    def model_dump(self, *args: Any, **kwargs: Any) -> Any:
        """Return dict representation."""
        return super().model_dump(*args, **kwargs)
