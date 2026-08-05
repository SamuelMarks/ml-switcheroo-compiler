# ruff: noqa: E501
"""Backend utilities."""

import tensorflow as tf

from ml_switcheroo_compiler.backends.eager import (
    generic_array,
    generic_asarray,
    generic_item,
    generic_zeros,
)


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
    return generic_zeros(tf, shape)


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
    return generic_array(tf, data, dtype)


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
    return generic_asarray(tf, data)


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
    return generic_item(tf, data)
