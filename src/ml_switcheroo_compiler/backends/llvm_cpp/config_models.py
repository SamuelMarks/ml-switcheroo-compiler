"""Pydantic models for C++ generator configuration."""

from typing import Optional

from pydantic import BaseModel, Field


class CppTilingConfig(BaseModel):
    """Configuration for CPU loop tiling.

    Attributes:
        tile_size_m (int): Tile size along M dimension.
        tile_size_n (int): Tile size along N dimension.
        tile_size_k (int): Tile size along K dimension.
    """

    tile_size_m: int = Field(default=16, description="Tile size along M dimension")
    tile_size_n: int = Field(default=16, description="Tile size along N dimension")
    tile_size_k: int = Field(default=16, description="Tile size along K dimension")


class CppSimdConfig(BaseModel):
    """Configuration for CPU SIMD pragmas and vectorization.

    Attributes:
        pragma (str): Compiler pragma for vectorization (e.g., #pragma omp simd).
        vector_width (int): Vector width in elements.
    """

    pragma: str = Field(default="#pragma omp simd", description="Compiler pragma for vectorization")
    vector_width: int = Field(default=8, description="Vector width in elements")


class CppTemplateConfig(BaseModel):
    """Configuration for a C++ template.

    Attributes:
        body (Optional[str]): Template code string.
        includes (Optional[list[str]]): Required C++ header includes.
        tiling (Optional[CppTilingConfig]): Loop tiling configuration.
        simd (Optional[CppSimdConfig]): SIMD configuration.
    """

    body: Optional[str] = None
    includes: Optional[list[str]] = None
    tiling: Optional[CppTilingConfig] = None
    simd: Optional[CppSimdConfig] = None
    model_config = {"extra": "allow"}


class CppOpConfig(BaseModel):
    """Configuration for a declarative C++ operation mapping.

    Attributes:
        template (str): Name of the compute template to apply (e.g., 'unary', 'binary').
        scalar_expr (str): Scalar C++ expression computing output value.
    """

    template: str = Field(description="Template name for operation")
    scalar_expr: str = Field(description="Scalar C++ compute expression")


class CppTemplatesConfig(BaseModel):
    """Configuration for all C++ templates.

    Attributes:
        prelude (Optional[str]): C++ prelude header declarations and helpers.
        templates (dict[str, CppTemplateConfig]): Map of operation names to C++ templates.
        operations (dict[str, CppOpConfig]): Declarative operation mappings to templates and expressions.
    """

    prelude: Optional[str] = None
    templates: dict[str, CppTemplateConfig] = Field(default_factory=dict)
    operations: dict[str, CppOpConfig] = Field(default_factory=dict)
