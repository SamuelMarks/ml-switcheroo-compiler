# ruff: noqa: E402
"""Provide a registry for Vector-Jacobian Product (VJP) rules used in reverse-mode automatic differentiation.

This module allows registering and retrieving VJP functions for various mathematical
operations, enabling the computation of gradients during the backward pass.
"""

from __future__ import annotations

import os
import typing
from collections.abc import Callable, Sequence

import yaml
from ml_switcheroo_ir import LogicalGraph, LogicalNode
from pydantic import BaseModel, Field

from ml_switcheroo_compiler.transforms.autodiff_rules.common import UnconnectedGradients


class VJPRuleCallable(typing.Protocol):
    """Protocol for Vector-Jacobian Product (VJP) rule callables."""

    def __call__(
        self,
        graph: LogicalGraph,
        node: LogicalNode,
        cotangent: str,
        /,
    ) -> Sequence[str | UnconnectedGradients | None] | str:
        """Evaluate the VJP rule for a node.

        Args:
            graph (LogicalGraph): Target IR graph.
            node (LogicalNode): Target IR node.
            cotangent (str): Incoming cotangent node ID.

        Returns:
            Sequence[str | UnconnectedGradients | None] | str: Input adjoints.
        """
        ...


class VJPRuleModel(BaseModel):
    """Declarative Vector-Jacobian Product (VJP) rule specification."""

    vjp: list[str] = Field(description="List of symbolic derivative expressions for each input operand.")
    description: str | None = Field(default=None, description="Human-readable description of the rule.")


class PrimitiveVJPsModel(BaseModel):
    """Container model for primitive VJP rules."""

    rules: dict[str, VJPRuleModel] = Field(default_factory=dict, description="Mapping of op_name to VJP rule.")


_DECLARATIVE_VJP_CACHE: dict[str, VJPRuleModel] | None = None


def load_primitive_vjp_rules(path: str | None = None) -> dict[str, VJPRuleModel]:
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
_VJP_REGISTRY: dict[str, VJPRuleCallable] = {}


def register_vjp(op_name: str) -> Callable[[VJPRuleCallable], VJPRuleCallable]:
    """Register a Vector-Jacobian Product (VJP) rule for a specific operation.

    Args:
        op_name (str): The op_name parameter.

    Returns:
        Callable[[VJPRuleCallable], VJPRuleCallable]: Decorator function.
    """

    def decorator(func: VJPRuleCallable) -> VJPRuleCallable:
        """Execute decorator for compiler processing.

        Args:
            func (VJPRuleCallable): The func parameter.

        Returns:
            VJPRuleCallable: Result.
        """
        if op_name in _VJP_REGISTRY:
            msg = f"VJP for operation '{op_name}' is already registered."
            raise ValueError(msg)
        _VJP_REGISTRY[op_name] = func
        return func

    return decorator


from ml_switcheroo_compiler.transforms.autodiff_rules.autodiff_provider import get_vjp_from_data


def get_vjp(op_name: str) -> VJPRuleCallable:
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
    if op_name in _VJP_REGISTRY:
        return _VJP_REGISTRY[op_name]
    raise ValueError(f"No VJP rule registered for operation: {op_name}")


def has_vjp(op_name: str) -> bool:
    """Check if a rule is registered.

    Args:
        op_name (str): The op_name parameter.

    Returns:
        bool: Result.
    """
    if op_name in _VJP_REGISTRY:
        return True
    return get_vjp_from_data(op_name) is not None
