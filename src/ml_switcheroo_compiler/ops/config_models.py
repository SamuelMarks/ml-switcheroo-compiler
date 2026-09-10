"""Pydantic models for ops registry configuration files."""

from collections.abc import ItemsView
from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, Field, RootModel


class VariantConfig(BaseModel):
    """Configuration for an op variant on a specific backend."""

    generator: Optional[str] = None
    eager: Optional[str] = None
    expr: Optional[str] = None
    scalar_expr: Optional[str] = None
    simd_expr: Optional[str] = None
    template: Optional[str] = None
    model_config = ConfigDict(extra="allow")


class OpArgConfig(BaseModel):
    """Configuration for an argument to an op."""

    name: str
    type: str
    is_variadic: Optional[bool] = False


class AutodiffConfig(BaseModel):
    """Configuration for autodiff rules."""

    jvp: Optional[str] = None
    vjp: Optional[list[str]] = None


class OpRegistryConfig(BaseModel):
    """Configuration for a specific op in the registry."""

    description: Optional[str] = None
    operation: Optional[str] = None
    std_args: Optional[list[Union[str, int, float, bool, OpArgConfig, dict[str, Union[str, int, float, bool]]]]] = None
    autodiff: Optional[AutodiffConfig] = None
    variants: dict[str, VariantConfig] = Field(default_factory=dict)
    model_config = ConfigDict(extra="allow")


class OpsRegistry(RootModel[dict[str, OpRegistryConfig]]):
    """Configuration for all ops in the registry."""

    root: dict[str, OpRegistryConfig]

    def items(self) -> ItemsView[str, OpRegistryConfig]:
        """Return items from the underlying dictionary.

        Returns:
            ItemsView[str, OpRegistryConfig]: Key-value pairs of the registry.
        """
        return self.root.items()

    def get(self, key: str, default: Optional[OpRegistryConfig] = None) -> Optional[OpRegistryConfig]:
        """Get op config by key.

        Args:
            key (str): Operation identifier.
            default (Optional[OpRegistryConfig]): Default fallback if not found.

        Returns:
            Optional[OpRegistryConfig]: Retrieved config or default.
        """
        return self.root.get(key, default)
