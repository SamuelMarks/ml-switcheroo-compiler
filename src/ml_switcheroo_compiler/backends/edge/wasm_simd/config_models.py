"""Pydantic models for WASM SIMD configuration."""

from typing import Any, Optional

from pydantic import BaseModel


class WasmTemplateConfig(BaseModel):
    """Configuration for a WASM template."""

    simd_unroll_factor: Optional[int] = None
    body: Optional[str] = None
    peel_loop: Optional[str] = None
    global_code: Optional[str] = None
    model_config = {"extra": "allow"}


class WasmTemplatesConfig(BaseModel):
    """Configuration for all WASM templates."""

    templates: dict[str, WasmTemplateConfig]
    js_orchestration: dict[str, str] = {}
    cpp_helpers: list[str] = []

    def model_dump(self, *args: Any, **kwargs: Any) -> Any:
        """Return dict representation."""
        res = super().model_dump(*args, **kwargs)
        return res
