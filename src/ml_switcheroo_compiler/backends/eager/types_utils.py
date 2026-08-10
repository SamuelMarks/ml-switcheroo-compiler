# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Types utils for eager backend."""

from typing import Any


def generic_zeros(mod: Any, shape: tuple) -> Any:
    """Provide generic zeros.

    Args:
        mod (object): The mod parameter.
        shape (tuple): The shape parameter.

    Returns: Any: Result.
    """
    return mod.zeros(shape)


def generic_array(mod: Any, data: Any, dtype: Any = None) -> Any:
    """Provide generic array.

    Args:
        mod (object): The mod parameter.
        data (object): The data parameter.
        dtype (object): The dtype parameter.

    Returns: Any: Result.
    """
    if dtype is not None:
        return mod.array(data, dtype=dtype)
    return mod.array(data)


def generic_asarray(mod: Any, data: Any) -> Any:
    """Provide generic asarray.

    Args:
        mod (object): The mod parameter.
        data (object): The data parameter.

    Returns: Any: Result.
    """
    return mod.asarray(data)


def generic_item(mod: Any, data: Any) -> Any:
    """Provide generic item.

    Args:
        mod (object): The mod parameter.
        data (object): The data parameter.

    Returns: Any: Result.
    """
    return data.item()
