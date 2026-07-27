# ruff: noqa: E501
"""Backend utilities."""

import keras.ops as kops

from ml_switcheroo_compiler.backends.eager import (
    generic_array,
    generic_asarray,
    generic_item,
    generic_zeros,
)


def zeros(cls: type, shape: tuple[int, ...]) -> object:
    """Execute zeros.

    Args:
        cls (Any): The cls parameter for the operation.
        shape (Any): Argument shape.

    Returns:
    Any: The result.
    """
    return generic_zeros(kops, shape)


def array(cls: type, data: object, dtype: object = None) -> object:
    """Execute array.

    Args:
        cls (Any): The cls parameter for the operation.
        data (Any): Argument data.
        dtype (Any): Argument dtype.

    Returns:
    Any: The result.
    """
    return generic_array(kops, data, dtype)


def asarray(cls: type, data: object) -> object:
    """Execute asarray.

    Args:
        cls (Any): The cls parameter for the operation.
        data (Any): Argument data.

    Returns:
    Any: The result.
    """
    return generic_asarray(kops, data)


def item(cls: type, data: object) -> float:
    """Execute item.

    Args:
        cls (Any): The cls parameter for the operation.
        data (Any): Argument data.

    Returns:
    Any: The result.
    """
    return generic_item(kops, data)
