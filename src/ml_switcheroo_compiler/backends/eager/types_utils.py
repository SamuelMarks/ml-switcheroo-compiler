# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Types utils for eager backend."""

from typing import Any, Optional


def generic_zeros(mod: object, shape: tuple[int, ...]) -> object:
    """Provide generic zeros.

    Args:
        mod: The mod parameter.
        shape: The shape parameter.

    Returns:
            object: Result.
    """
    return mod.zeros(shape)


def generic_array(mod: object, data: object, dtype: Optional[object] = None) -> object:
    """Provide generic array.

    Args:
        mod: The mod parameter.
        data: The data parameter.
        dtype: The dtype parameter.

    Returns:
            object: Result.
    """
    if dtype is not None:
        return mod.array(data, dtype=dtype)
    return mod.array(data)


def generic_asarray(mod: object, data: object) -> object:
    """Provide generic asarray.

    Args:
        mod: The mod parameter.
        data: The data parameter.

    Returns:
            object: Result.
    """
    return mod.asarray(data)


def generic_item(mod: object, data: object) -> float:
    """Provide generic item.

    Args:
        mod: The mod parameter.
        data: The data parameter.

    Returns:
            float: Result.
    """
    return float(data.item())
