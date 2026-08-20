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
        """Dump the model."""
        """Return dict representation."""
        res = super().model_dump(*args, **kwargs)
        return res


class WasmIntrinsicConfig(BaseModel):
    """Configuration for a WASM intrinsic."""

    macro_name: str
    simd_expr: str
    scalar_fallback: str


class WasmIntrinsicsConfig(BaseModel):
    """Configuration for all WASM intrinsics."""

    intrinsics: dict[str, WasmIntrinsicConfig]
    scalars: dict[str, str]

    def model_dump(self, *args: Any, **kwargs: Any) -> Any:
        """Dump the model."""
        return super().model_dump(*args, **kwargs)
