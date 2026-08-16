"""Pydantic models for ops registry configuration files."""

from typing import Any, Optional

from pydantic import BaseModel, Field, RootModel


class VariantConfig(BaseModel):
    """Configuration for an op variant on a specific backend."""

    generator: Optional[str] = None
    eager: Optional[str] = None
    expr: Optional[str] = None
    scalar_expr: Optional[str] = None
    simd_expr: Optional[str] = None
    template: Optional[str] = None
    model_config = {"extra": "allow"}  # Allow other backend specific configurations


class OpArgConfig(BaseModel):
    """Configuration for an argument to an op."""

    name: str
    type: str
    is_variadic: Optional[bool] = False


class OpRegistryConfig(BaseModel):
    """Configuration for a specific op in the registry."""

    description: Optional[str] = None
    operation: Optional[str] = None
    std_args: Optional[list[Any]] = None  # Just let it be Any for now, sometimes it's dict, sometimes string
    variants: dict[str, VariantConfig] = Field(default_factory=dict)
    model_config = {"extra": "allow"}


class OpsRegistry(RootModel[dict[str, OpRegistryConfig]]):
    """Configuration for all ops in the registry."""

    root: dict[str, OpRegistryConfig]

    def dict(self, *args: Any, **kwargs: Any) -> Any:
        """Return dict representation."""
        return super().model_dump(*args, **kwargs)

    def items(self) -> Any:
        """Return items from the underlying dictionary."""
        return self.root.items()

    def get(self, key: str, default: Any = None) -> Any:
        """Get op config by key."""
        return self.root.get(key, default)
