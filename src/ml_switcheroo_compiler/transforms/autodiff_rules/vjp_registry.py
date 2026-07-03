"""Provides a registry for Vector-Jacobian Product (VJP) rules used in reverse-mode.

automatic differentiation

This module allows registering and retrieving VJP functions for various mathematical
operations, enabling the computation of gradients during the backward pass
"""

from typing import Callable

from ml_switcheroo_compiler.core.errors import UnimplementedMathError

# Registry mapping op_name to VJP function
_VJP_REGISTRY: dict[str, Callable] = {}


def register_vjp(op_name: str) -> Callable:
    """Registers a Vector-Jacobian Product (VJP) rule for a specific operation.

    This function acts as a decorator factory. The decorated function should
    implement the VJP for the specified operation

    Args:
        op_name (str): The name of the operation (e.g., 'add', 'multiply')

    Returns:
    Callable: A decorator function that registers the VJP rule

    Raises:
    ValueError: If a VJP rule is already registered for the given `op_name` when the
    decorator is applied
    """

    def decorator(func: Callable) -> Callable:
        """Execute decorator.

        Args:
            func (Any): Argument func.

        Returns:
        Any: The result.
        """
        if op_name in _VJP_REGISTRY:
            msg = f"VJP for operation '{op_name}' is already registered."
            raise ValueError(msg)
        _VJP_REGISTRY[op_name] = func
        return func

    return decorator


def get_vjp(op_name: str) -> Callable:
    """Retrieves the registered Vector-Jacobian Product (VJP) rule for a given operation.

    Args:
        op_name (str): The name of the operation whose VJP rule is to be retrieved

    Returns:
    Callable: The registered VJP function for the operation

    Raises:
    UnimplementedMathError: If no VJP rule has been registered for the specified
    operation
    """
    if op_name not in _VJP_REGISTRY:
        msg = f"VJP not implemented for {op_name}"
        raise UnimplementedMathError(msg)
    return _VJP_REGISTRY[op_name]
