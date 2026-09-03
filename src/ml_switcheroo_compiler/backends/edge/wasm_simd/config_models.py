"""Pydantic models for WASM SIMD configuration."""

from typing import Optional

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

    def model_dump(self, *args: object, **kwargs: object) -> object:
        """Dump the model.

        Return dict representation.
        """
        res = super().model_dump(*args, **kwargs)
        return res


class WasmIntrinsicConfig(BaseModel):
    """Configuration for a WASM intrinsic."""

    macro_name: Optional[str] = None
    simd_expr: Optional[str] = None
    scalar_fallback: Optional[str] = None


class WasmIntrinsicsConfig(BaseModel):
    """Configuration for all WASM intrinsics."""

    intrinsics: dict[str, WasmIntrinsicConfig]
    scalars: Optional[dict[str, str]] = None

    def model_dump(self, *args: object, **kwargs: object) -> object:
        """Dump the model."""
        return super().model_dump(*args, **kwargs)
