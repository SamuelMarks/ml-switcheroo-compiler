# ruff: noqa: E402, E501, F403
"""Backend utilities."""

from typing import Optional

import jax.numpy as jnp

from ml_switcheroo_compiler.backends.eager import (
    generic_array,
    generic_asarray,
    generic_item,
    generic_zeros,
)


def zeros(cls: type, shape: tuple[int, ...]) -> object:
    """Evaluate zeros operation.

    Args:
        cls: Class.
        shape (tuple[int, ...]): Shape.

    Returns:
            object: Result.
    """
    return generic_zeros(jnp, shape)


def array(cls: type, data: object, dtype: Optional[object] = None) -> object:
    """Evaluate array operation.

    Args:
        cls: Class.
        data: Data.
        dtype: Dtype.

    Returns:
            object: Result.
    """
    return generic_array(jnp, data, dtype)


def asarray(cls: type, data: object) -> object:
    """Evaluate asarray operation.

    Args:
        cls: Class.
        data: Data.

    Returns:
            object: Result.
    """
    return generic_asarray(jnp, data)


def item(cls: type, data: object) -> float:
    """Evaluate item operation.

    Args:
        cls: Class.
        data: Data.

    Returns:
        float: Result.
    """
    return generic_item(jnp, data)
