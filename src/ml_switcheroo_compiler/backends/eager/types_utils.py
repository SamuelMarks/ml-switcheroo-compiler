"""Types utils for eager backend."""


def generic_zeros(mod: object, shape: tuple) -> object:
    """Provide generic zeros.

    Args:
        mod (object): The mod parameter.
        shape (tuple): The shape parameter.

    Returns:
        object: Result.
    """
    return mod.zeros(shape)


def generic_array(mod: object, data: object, dtype: object = None) -> object:
    """Provide generic array.

    Args:
        mod (object): The mod parameter.
        data (object): The data parameter.
        dtype (object): The dtype parameter.

    Returns:
        object: Result.
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
        object: Result.
    """
    return mod.asarray(data)


def generic_item(mod: object, data: object) -> object:
    """Provide generic item.

    Args:
        mod (object): The mod parameter.
        data (object): The data parameter.

    Returns:
        object: Result.
    """
    return data.item()
