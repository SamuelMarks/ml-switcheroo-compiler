# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
from typing import Any

"""Provide a registry for Vector-Jacobian Product (VJP) rules used in reverse-mode automatic differentiation.

This module allows registering and retrieving VJP functions for various mathematical
operations, enabling the computation of gradients during the backward pass.
"""

from typing import Callable

# Registry mapping op_name to VJP function
_VJP_REGISTRY: dict[str, Callable] = {}


def register_vjp(op_name: str) -> Callable:
    """Register a Vector-Jacobian Product (VJP) rule for a specific operation.

    Args:
        op_name (str): The op_name parameter.

    Returns:
        Callable: Result.
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
    if get_vjp_from_data(op_name) is not None:
        return True
    return op_name in _VJP_REGISTRY
