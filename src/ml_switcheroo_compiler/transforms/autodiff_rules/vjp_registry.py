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


def get_vjp(op_name: str) -> Callable:
    """Retrieve the registered Vector-Jacobian Product (VJP) rule for a given operation.

    Args:
        op_name (str): The op_name parameter.

    Returns:
        Callable: Result.
    """
    if op_name not in _VJP_REGISTRY:
        raise ValueError(f"Missing VJP rule for operation: {op_name}")
    return _VJP_REGISTRY[op_name]
