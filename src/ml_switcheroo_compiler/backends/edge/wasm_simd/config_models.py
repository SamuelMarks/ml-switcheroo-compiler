"""Pydantic models for WASM SIMD configuration."""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict


class MemoryAlignmentConfig(BaseModel):
    """Memory alignment configuration for SIMD vector loads/stores.

    Attributes:
        alignment_bytes (int): Alignment boundary in bytes (e.g. 16 for v128).
        requires_aligned_load (bool): Whether vector load requires aligned pointer.
        peeling_loop_required (bool): Whether prefix/suffix peeling loops are needed.
    """

    model_config = ConfigDict(extra="allow")

    alignment_bytes: int = 16
    requires_aligned_load: bool = True
    peeling_loop_required: bool = True


class LaneArithmeticConfig(BaseModel):
    """Configuration for lane-wise SIMD arithmetic operations.

    Attributes:
        lanes (int): Number of parallel vector lanes (e.g. 4 for f32x4).
        element_type (str): Element primitive type identifier.
        simd_intrinsic (str): Underlying hardware vector intrinsic name.
        lane_mask (Optional[str]): Optional bitmask for conditional lane operations.
    """

    model_config = ConfigDict(extra="allow")

    lanes: int = 4
    element_type: Literal["f32", "i32", "i16", "i8", "f64"] = "f32"
    simd_intrinsic: str = ""
    lane_mask: Optional[str] = None


class WasmTemplateConfig(BaseModel):
    """Configuration for a WASM template.

    Attributes:
        simd_unroll_factor (Optional[int]): Unroll factor for SIMD vector loops.
        body (Optional[str]): Main kernel C++/SIMD body template.
        peel_loop (Optional[str]): Scalar fallback loop for unaligned or remainder elements.
        global_code (Optional[str]): Supporting global declarations or static helper code.
        alignment (Optional[MemoryAlignmentConfig]): Vector memory alignment configuration.
        lane_arithmetic (Optional[LaneArithmeticConfig]): Lane-wise arithmetic specifications.
    """

    model_config = ConfigDict(extra="allow")

    simd_unroll_factor: Optional[int] = None
    body: Optional[str] = None
    peel_loop: Optional[str] = None
    global_code: Optional[str] = None
    alignment: Optional[MemoryAlignmentConfig] = None
    lane_arithmetic: Optional[LaneArithmeticConfig] = None


class WasmTemplatesConfig(BaseModel):
    """Configuration for all WASM templates.

    Attributes:
        templates (dict[str, WasmTemplateConfig]): Map of operation names to template configs.
        js_orchestration (dict[str, str]): Map of orchestration names to JS snippets.
        cpp_helpers (list[str]): List of reusable C++ header helpers.
    """

    model_config = ConfigDict(extra="allow")

    templates: dict[str, WasmTemplateConfig]
    js_orchestration: dict[str, str] = {}
    cpp_helpers: list[str] = []


class WasmIntrinsicConfig(BaseModel):
    """Configuration for a WASM intrinsic.

    Attributes:
        macro_name (Optional[str]): Macro identifier emitted for the intrinsic.
        simd_expr (Optional[str]): Body of the vector intrinsic expression.
        scalar_fallback (Optional[str]): Scalar fallback expression for non-vector paths.
        alignment (Optional[MemoryAlignmentConfig]): Vector memory alignment parameters.
        lane_arithmetic (Optional[LaneArithmeticConfig]): Lane arithmetic specifications.
    """

    model_config = ConfigDict(extra="allow")

    macro_name: Optional[str] = None
    simd_expr: Optional[str] = None
    scalar_fallback: Optional[str] = None
    alignment: Optional[MemoryAlignmentConfig] = None
    lane_arithmetic: Optional[LaneArithmeticConfig] = None


class WasmIntrinsicsConfig(BaseModel):
    """Configuration for all WASM intrinsics.

    Attributes:
        intrinsics (dict[str, WasmIntrinsicConfig]): Map of op names to intrinsic configs.
        scalars (Optional[dict[str, str]]): Map of scalar fallback functions.
    """

    model_config = ConfigDict(extra="allow")

    intrinsics: dict[str, WasmIntrinsicConfig]
    scalars: Optional[dict[str, str]] = None
