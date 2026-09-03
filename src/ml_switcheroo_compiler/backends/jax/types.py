# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
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
