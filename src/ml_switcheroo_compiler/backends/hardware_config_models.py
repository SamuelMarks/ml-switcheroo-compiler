"""Pydantic models for hardware generators."""

from typing import Any

from pydantic import BaseModel


class HardwareTemplateConfig(BaseModel):
    """Configuration for a hardware template."""

    body: str
    workgroup_size: list[int] = [256, 1, 1]


class HardwareTemplatesConfig(BaseModel):
    """Configuration for all hardware templates."""

    templates: dict[str, HardwareTemplateConfig]
    orchestration: dict[str, str] = {}

    def model_dump(self, *args: Any, **kwargs: Any) -> Any:
        """Dump the model."""
        return super().model_dump(*args, **kwargs)
