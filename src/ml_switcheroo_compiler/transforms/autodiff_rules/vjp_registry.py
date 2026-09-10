# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Provide a registry for Vector-Jacobian Product (VJP) rules used in reverse-mode automatic differentiation.

This module allows registering and retrieving VJP functions for various mathematical
operations, enabling the computation of gradients during the backward pass.
"""

import os
from collections.abc import Callable
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class VJPRuleModel(BaseModel):
    """Declarative Vector-Jacobian Product (VJP) rule specification."""

    vjp: list[str] = Field(description="List of symbolic derivative expressions for each input operand.")
    description: Optional[str] = Field(default=None, description="Human-readable description of the rule.")


class PrimitiveVJPsModel(BaseModel):
    """Container model for primitive VJP rules."""

    rules: dict[str, VJPRuleModel] = Field(default_factory=dict, description="Mapping of op_name to VJP rule.")


_DECLARATIVE_VJP_CACHE: Optional[dict[str, VJPRuleModel]] = None


def load_primitive_vjp_rules(path: Optional[str] = None) -> dict[str, VJPRuleModel]:
    """Load and validate declarative primitive VJP rules via Pydantic model.

    Args:
        path (Optional[str]): Path to primitive VJPs YAML file.

    Returns:
        dict[str, VJPRuleModel]: Parsed and validated VJP rule specifications.
    """
    global _DECLARATIVE_VJP_CACHE
    if _DECLARATIVE_VJP_CACHE is not None and path is None:
        return _DECLARATIVE_VJP_CACHE

    target_path = path or os.path.join(os.path.dirname(__file__), "primitive_vjps.yaml")
    if not os.path.exists(target_path):
        target_path = os.path.join(os.path.dirname(__file__), "vjp_rules.yaml")

    loaded_rules: dict[str, VJPRuleModel] = {}
    if os.path.exists(target_path):
        with open(target_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, dict) and "vjp" in v:
                    loaded_rules[k] = VJPRuleModel(**v)

    if path is None:
        _DECLARATIVE_VJP_CACHE = loaded_rules
    return loaded_rules


# Registry mapping op_name to VJP function
_VJP_REGISTRY: dict[str, Callable] = {}


def register_vjp(op_name: str) -> Callable[[Callable], Callable]:
    """Register a Vector-Jacobian Product (VJP) rule for a specific operation.

    Args:
        op_name (str): The op_name parameter.

    Returns:
        Callable[[Callable], Callable]: Decorator function.
    """

    def decorator(func: Callable) -> Callable:
        """Evaluate decorator operation.

        Args:
            func (Callable): The func parameter.

        Returns:
            Callable: Result.
        """
        if op_name in _VJP_REGISTRY:
            msg = f"VJP for operation '{op_name}' is already registered."
            raise ValueError(msg)
        _VJP_REGISTRY[op_name] = func
        return func

    return decorator


from ml_switcheroo_compiler.transforms.autodiff_rules.autodiff_provider import get_vjp_from_data


def get_vjp(op_name: str) -> Callable:
    """Get the VJP rule.

    Args:
        op_name (str): The op_name parameter.

    Returns:
        Callable: Result.

    Raises:
        ValueError: An exception.
    """
    if op_name in _VJP_REGISTRY:
        return _VJP_REGISTRY[op_name]
    data_vjp = get_vjp_from_data(op_name)
    if data_vjp:
        return data_vjp
    if op_name not in _VJP_REGISTRY:
        raise ValueError(f"No VJP rule registered for operation: {op_name}")
    return _VJP_REGISTRY[op_name]


def has_vjp(op_name: str) -> bool:
    """Check if a rule is registered.

    Args:
        op_name (str): The op_name parameter.

    Returns:
        bool: Result.
    """
    if op_name in _VJP_REGISTRY:
        return True
    if get_vjp_from_data(op_name) is not None:
        return True
    return op_name in _VJP_REGISTRY
