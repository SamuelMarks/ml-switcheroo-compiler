# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Types utils for eager backend."""

from typing import Any


def generic_zeros(mod: Any, shape: Any) -> Any:
    """Provide generic zeros.

    Args:
        mod (Any): The mod parameter.
        shape (Any): The shape parameter.

    Returns:
            Any: Result.
    """
    return mod.zeros(shape)


def generic_array(mod: Any, data: Any, dtype: Any = None) -> Any:
    """Provide generic array.

    Args:
        mod (Any): The mod parameter.
        data (Any): The data parameter.
        dtype (Any): The dtype parameter.

    Returns:
            Any: Result.
    """
    if dtype is not None:
        return mod.array(data, dtype=dtype)
    return mod.array(data)


def generic_asarray(mod: Any, data: Any) -> Any:
    """Provide generic asarray.

    Args:
        mod (Any): The mod parameter.
        data (Any): The data parameter.

    Returns:
            Any: Result.
    """
    return mod.asarray(data)


def generic_item(mod: Any, data: Any) -> float:
    """Provide generic item.

    Args:
        mod (Any): The mod parameter.
        data (Any): The data parameter.

    Returns:
            float: Result.
    """
    return float(data.item())
