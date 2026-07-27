"""Provides a registry for Jacobian-Vector Product (JVP) rules used in forward-mode.

automatic differentiation

This module allows registering and retrieving JVP functions for various mathematical
operations
"""

from typing import Callable

# Registry mapping op_name to JVP function
_JVP_REGISTRY: dict[str, Callable] = {}


def register_jvp(op_name: str) -> Callable:
    """Registers a Jacobian-Vector Product (JVP) rule for a specific mathematical.

    operation

    This function acts as a decorator factory. The decorated function should define
    how to compute the JVP for the given operation name

    Args:
        op_name (str): The unique name of the operation to register

    Returns:
    Callable: A decorator function that registers the decorated JVP rule

    Raises:
    ValueError: If a JVP rule for the specified operation name is already
    registered
    """

    def decorator(func: Callable) -> Callable:
        """Execute decorator.

        Args:
            func (Any): Argument func.

        Returns:
        Any: The result.
        """
        if op_name in _JVP_REGISTRY:
            msg = f"JVP for operation '{op_name}' is already registered."
            raise ValueError(msg)
        _JVP_REGISTRY[op_name] = func
        return func

    return decorator


def get_jvp(op_name: str) -> Callable:
    """Retrieves the registered Jacobian-Vector Product (JVP) rule for a given operation.

    Args:
        op_name (str): The name of the operation whose JVP rule is being requested

    Returns:
    Callable: The registered JVP function associated with the operation

    Raises:
    ValueError: If no JVP rule has been registered for the specified
    operation
    """
    if op_name not in _JVP_REGISTRY:
        raise ValueError(f"Missing JVP rule for operation: {op_name}")
    return _JVP_REGISTRY[op_name]
