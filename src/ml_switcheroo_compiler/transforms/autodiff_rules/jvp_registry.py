"""Provide a registry for Jacobian-Vector Product (JVP) rules used in forward-mode automatic differentiation.

This module allows registering and retrieving JVP functions for various mathematical
operations.
"""

from typing import Callable

# Registry mapping op_name to JVP function
_JVP_REGISTRY: dict[str, Callable] = {}


def register_jvp(op_name: str) -> Callable:
    """Register a Jacobian-Vector Product (JVP) rule for a specific mathematical.

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
        if op_name in _JVP_REGISTRY:
            msg = f"JVP for operation '{op_name}' is already registered."
            raise ValueError(msg)
        _JVP_REGISTRY[op_name] = func
        return func

    return decorator


def get_jvp(op_name: str) -> Callable:
    """Retrieve the registered Jacobian-Vector Product (JVP) rule for a given operation.

    Args:
        op_name (str): The op_name parameter.

    Returns:
        Callable: Result.
    """
    if op_name not in _JVP_REGISTRY:
        raise ValueError(f"Missing JVP rule for operation: {op_name}")
    return _JVP_REGISTRY[op_name]
