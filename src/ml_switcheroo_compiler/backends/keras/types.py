"""Backend utilities."""

import keras.ops as kops

from ml_switcheroo_compiler.backends.eager_utils import (
    generic_array,
    generic_asarray,
    generic_item,
    generic_zeros,
)


def zeros(cls: type, shape: tuple[int, ...]) -> object:
    """Execute zeros.

    Args:
        cls (Any): The class.
        shape (Any): Argument shape.

    Returns:
    Any: The result.
    """
    return generic_zeros(kops, shape)


def array(cls: type, data: object) -> object:
    """Execute array.

    Args:
        cls (Any): The class.
        data (Any): Argument data.

    Returns:
    Any: The result.
    """
    return generic_array(kops, data)


def asarray(cls: type, data: object) -> object:
    """Execute asarray.

    Args:
        cls (Any): The class.
        data (Any): Argument data.

    Returns:
    Any: The result.
    """
    return generic_asarray(kops, data)


def item(cls: type, data: object) -> float:
    """Execute item.

    Args:
        cls (Any): The class.
        data (Any): Argument data.

    Returns:
    Any: The result.
    """
    return generic_item(kops, data)
