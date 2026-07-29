"""Types utils for eager backend."""


def generic_zeros(mod: object, shape: tuple) -> object:
    """Generic zeros."""
    return mod.zeros(shape)


def generic_array(mod: object, data: object, dtype: object = None) -> object:
    """Generic array."""
    if dtype is not None:
        return mod.array(data, dtype=dtype)
    return mod.array(data)


def generic_asarray(mod: object, data: object) -> object:
    """Generic asarray."""
    return mod.asarray(data)


def generic_item(mod: object, data: object) -> object:
    """Generic item."""
    return data.item()
