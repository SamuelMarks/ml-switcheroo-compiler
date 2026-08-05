# ruff: noqa: E501
"""Backend utilities."""

import torch


def zeros(cls: type, shape: tuple[int, ...]) -> object:
    """Evaluate zeros operation.

    Args:
        cls (type): Class.
        shape (tuple): Shape.

    Args:
        cls (type): Class.
        shape (tuple): Shape.

    Args:
        cls (type): The class.
    shape (tuple): The shape parameter.

    Returns:
        object: Result.
    """
    return torch.zeros(shape)


def array(cls: type, data: object, dtype: object = None) -> object:
    """Evaluate array operation.

    Args:
        cls (type): Class.
        data (object): Data.
        dtype (object): Dtype.

    Args:
        cls (type): Class.
        data (object): Data.
        dtype (object): Dtype.

    Args:
        cls (type): The class.
    data (object): The data parameter.
        dtype (object): The dtype parameter.

    Returns:
        object: Result.
    """
    if dtype is not None:
        dtype_str = str(getattr(dtype, "value", dtype)).split(".")[-1]
        dt = getattr(torch, dtype_str, None)
        return torch.tensor(data, dtype=dt)
    return torch.tensor(data)


def asarray(cls: type, data: object) -> object:
    """Evaluate asarray operation.

    Args:
        cls (type): Class.
        data (object): Data.

    Args:
        cls (type): Class.
        data (object): Data.

    Args:
        cls (type): The class.
    data (object): The data parameter.

    Returns:
        object: Result.
    """
    return torch.as_tensor(data)


def item(cls: type, data: object) -> float:
    """Evaluate item operation.

    Args:
        cls (type): Class.
        data (object): Data.

    Args:
        cls (type): Class.
        data (object): Data.

    Args:
        cls (type): The class.
    data (object): The data parameter.

    Returns:
        float: Result.
    """
    return torch.as_tensor(data).item()
