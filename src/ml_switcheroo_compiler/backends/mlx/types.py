# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Backend utilities."""

from typing import Any

import mlx.core as mx

from ml_switcheroo_compiler.backends.eager import (
    generic_array,
    generic_asarray,
    generic_item,
    generic_zeros,
)


def zeros(cls: type, shape: tuple[int, ...]) -> Any:
    """Evaluate zeros operation.

    Args:
        cls (type): Class.
        shape (tuple[int, ...]): Shape.

    Returns:
            Any: Result.
    """
    return generic_zeros(mx, shape)


def array(cls: type, data: Any, dtype: Any = None) -> Any:
    """Evaluate array operation.

    Args:
        cls (type): Class.
        data (Any): Data.
        dtype (Any): Dtype.

    Returns:
            Any: Result.
    """
    return generic_array(mx, data, dtype)


def asarray(cls: type, data: Any) -> Any:
    """Evaluate asarray operation.

    Args:
        cls (type): Class.
        data (Any): Data.

    Returns:
            Any: Result.
    """
    if hasattr(mx, "asarray"):
        return generic_asarray(mx, data)
    return generic_array(mx, data)


def item(cls: type, data: Any) -> float:
    """Evaluate item operation.

    Args:
        cls (type): Class.
        data (Any): Data.

    Returns:
        float: Result.
    """
    return generic_item(mx, data)
