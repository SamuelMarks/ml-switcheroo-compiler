# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Module jvp_registry.py."""

"""Provide a registry for Jacobian-Vector Product (JVP) rules used in forward-mode automatic differentiation.

This module allows registering and retrieving JVP functions for various mathematical
operations.
"""

from typing import Callable

# Registry mapping op_name to JVP function
_JVP_REGISTRY: dict[str, Callable[..., object]] = {}


def register_jvp(op_name: str) -> Callable[..., object]:
    """Register a Jacobian-Vector Product (JVP) rule for a specific mathematical.

    Args:
        op_name (str): The op_name parameter.

    Returns:
        Callable: Result.
    """

    def decorator(func: Callable[..., object]) -> Callable[..., object]:
        """Evaluate decorator operation.

        Args:
            func (Callable): The func parameter.

        Returns:
            Callable: Result.
        """
        if op_name in _JVP_REGISTRY:
            msg: object = f"JVP for operation '{op_name}' is already registered."
            raise ValueError(msg)
        _JVP_REGISTRY[op_name] = func
        return func

    return decorator


from ml_switcheroo_compiler.transforms.autodiff_rules.autodiff_provider import get_jvp_from_data


def get_jvp(op_name: str) -> Callable[..., object]:
    """Get the JVP rule.

    Args:
        op_name (str): The op_name parameter.

    Returns:
        Callable: Result.

    Raises:
        ValueError: An exception.
    """
    data_jvp: object = get_jvp_from_data(op_name)
    if data_jvp:
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
    if get_jvp_from_data(op_name) is not None:
        return True
    return op_name in _JVP_REGISTRY
