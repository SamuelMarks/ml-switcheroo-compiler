"""Pydantic models for edge generator configuration."""

from typing import Any, Optional

from pydantic import BaseModel


class WgslTemplateConfig(BaseModel):
    """Configuration for a WGSL template."""

    workgroup_size: Optional[list[int]] = None
    body: Optional[str] = None
    global_code: Optional[str] = None
    model_config = {"extra": "allow"}


class WgslTemplatesConfig(BaseModel):
    """Configuration for all WGSL templates."""

    templates: dict[str, WgslTemplateConfig]

    def model_dump(self, *args: Any, **kwargs: Any) -> Any:
        """Return dict representation."""
        res = super().model_dump(*args, **kwargs)
        return res["templates"]


class MemoryLimitsConfig(BaseModel):
    """Configuration for edge memory limits."""

    max_arenas: int = 16
    arena_size_bytes: int = 134217728
    reuse_policy: str = "greedy"
