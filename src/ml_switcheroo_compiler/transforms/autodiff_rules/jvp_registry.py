# ruff: noqa: E402
"""Provide a registry for Jacobian-Vector Product (JVP) rules used in forward-mode automatic differentiation.

This module allows registering and retrieving JVP functions for various mathematical
operations.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Optional, Union

import yaml
from pydantic import BaseModel, Field

from ml_switcheroo_compiler.transforms.autodiff_rules.common import UnconnectedGradients

JVPRuleCallable = Callable[..., Union[str, tuple[str, ...], UnconnectedGradients, None]]


class JVPRuleModel(BaseModel):
    """Declarative Jacobian-Vector Product (JVP) rule specification."""

    jvp: str = Field(description="Symbolic derivative expression for the forward tangent propagation.")
    description: Optional[str] = Field(default=None, description="Human-readable description of the rule.")


class PrimitiveJVPsModel(BaseModel):
    """Container model for primitive JVP rules."""

    rules: dict[str, JVPRuleModel] = Field(default_factory=dict, description="Mapping of op_name to JVP rule.")


_DECLARATIVE_JVP_CACHE: Optional[dict[str, JVPRuleModel]] = None


def load_primitive_jvp_rules(path: Optional[str] = None) -> dict[str, JVPRuleModel]:
    """Load and validate declarative primitive JVP rules via Pydantic model.

    Args:
        path (Optional[str]): Path to primitive JVPs YAML file.

    Returns:
        dict[str, JVPRuleModel]: Parsed and validated JVP rule specifications.
    """
    global _DECLARATIVE_JVP_CACHE
    if _DECLARATIVE_JVP_CACHE is not None and path is None:
        return _DECLARATIVE_JVP_CACHE

    target_path = path or os.path.join(os.path.dirname(__file__), "primitive_jvps.yaml")
    if not os.path.exists(target_path):
        target_path = os.path.join(os.path.dirname(__file__), "jvp_rules.yaml")

    loaded_rules: dict[str, JVPRuleModel] = {}
    if os.path.exists(target_path):
        with open(target_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, dict) and "jvp" in v:
                    loaded_rules[k] = JVPRuleModel(**v)

    if path is None:
        _DECLARATIVE_JVP_CACHE = loaded_rules
    return loaded_rules


# Registry mapping op_name to JVP function
_JVP_REGISTRY: dict[str, JVPRuleCallable] = {}


def register_jvp(op_name: str) -> Callable[[JVPRuleCallable], JVPRuleCallable]:
    """Register a Jacobian-Vector Product (JVP) rule for a specific mathematical operation.

    Args:
        op_name (str): The op_name parameter.

    Returns:
        Callable[[JVPRuleCallable], JVPRuleCallable]: Decorator function.
    """

    def decorator(func: JVPRuleCallable) -> JVPRuleCallable:
        """Evaluate decorator operation.

        Args:
            func (JVPRuleCallable): The func parameter.

        Returns:
            JVPRuleCallable: Result.
        """
        if op_name in _JVP_REGISTRY:
            msg = f"JVP for operation '{op_name}' is already registered."
            raise ValueError(msg)
        _JVP_REGISTRY[op_name] = func
        return func

    return decorator


from ml_switcheroo_compiler.transforms.autodiff_rules.autodiff_provider import get_jvp_from_data


def get_jvp(op_name: str) -> JVPRuleCallable:
    """Get the JVP rule.

    Args:
        op_name (str): The op_name parameter.

    Returns:
        Callable: Result.

    Raises:
        ValueError: An exception.
    """
    if op_name in _JVP_REGISTRY:
        return _JVP_REGISTRY[op_name]
    data_jvp = get_jvp_from_data(op_name)
    if data_jvp is not None:
        return data_jvp
    if op_name not in _JVP_REGISTRY:
        raise ValueError(f"No JVP rule registered for operation: {op_name}")
    return _JVP_REGISTRY[op_name]


def has_jvp(op_name: str) -> bool:
    """Check if a rule is registered.

    Args:
        op_name (str): The op_name parameter.

    Returns:
        bool: Result.
    """
    if op_name in _JVP_REGISTRY:
        return True
    if get_jvp_from_data(op_name) is not None:
        return True
    return op_name in _JVP_REGISTRY
