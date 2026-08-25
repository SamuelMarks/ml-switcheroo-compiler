# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Types utils for eager backend."""


def generic_zeros(mod: object, shape: tuple[object, ...]) -> object:
    """Provide generic zeros.

    Args:
        mod (object): The mod parameter.
        shape (tuple): The shape parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return mod.zeros(shape)


def generic_array(mod: object, data: object, dtype: object = None) -> object:
    """Provide generic array.

    Args:
        mod (object): The mod parameter.
        data (object): The data parameter.
        dtype (object): The dtype parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    if dtype is not None:
        return mod.array(data, dtype=dtype)
    return mod.array(data)


def generic_asarray(mod: object, data: object) -> object:
    """Provide generic asarray.

    Args:
        mod (object): The mod parameter.
        data (object): The data parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return mod.asarray(data)


def generic_item(mod: object, data: object) -> object:
    """Provide generic item.

    Args:
        mod (object): The mod parameter.
        data (object): The data parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return data.item()
