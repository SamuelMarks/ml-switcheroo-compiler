# ruff: noqa: E501
"""Backend utilities."""

import torch


def zeros(cls: type, shape: tuple[int, ...]) -> object:
    """Execute zeros.

    Args:
        cls (Any): The cls parameter for the operation.
        shape (Any): Argument shape.

    Returns:
    Any: The result.
    """
    return torch.zeros(shape)


def array(cls: type, data: object, dtype: object = None) -> object:
    """Execute array.

    Args:
        cls (Any): The cls parameter for the operation.
        data (Any): Argument data.
        dtype (Any): Argument dtype.

    Returns:
    Any: The result.
    """
    if dtype is not None:
        dtype_str = str(getattr(dtype, "value", dtype)).split(".")[-1]
        dt = getattr(torch, dtype_str, None)
        return torch.tensor(data, dtype=dt)
    return torch.tensor(data)


def asarray(cls: type, data: object) -> object:
    """Execute asarray.

    Args:
        cls (Any): The cls parameter for the operation.
        data (Any): Argument data.

    Returns:
    Any: The result.
    """
    return torch.as_tensor(data)


def item(cls: type, data: object) -> float:
    """Execute item.

    Args:
        cls (Any): The cls parameter for the operation.
        data (Any): Argument data.

    Returns:
    Any: The result.
    """
    return torch.as_tensor(data).item()
