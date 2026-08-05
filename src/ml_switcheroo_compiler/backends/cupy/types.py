# ruff: noqa: E501
"""Backend utilities."""

from ml_switcheroo_compiler.backends.eager import (
    generic_array,
    generic_asarray,
    generic_item,
    generic_zeros,
)

try:
    import cupy as cp
except ImportError:
    cp = None


def zeros(cls: type, shape: tuple[int, ...]) -> object:
    """Create a tensor of zeros.

    Args:
        cls (type): The backend class.
        shape (tuple[int, ...]): The shape of the tensor.

    Returns:
        object: The zeros tensor.
    """
    return generic_zeros(cp, shape)


def array(cls: type, data: object, dtype: object = None) -> object:
    """Create a tensor from data.

    Args:
        cls (type): The backend class.
        data (object): The input data.
        dtype (object): The target dtype.

    Returns:
        object: The array tensor.
    """
    return generic_array(cp, data, dtype)


def asarray(cls: type, data: object) -> object:
    """Convert data to an array.

    Args:
        cls (type): The backend class.
        data (object): The input data.

    Returns:
        object: The array tensor.
    """
    return generic_asarray(cp, data)


def item(cls: type, data: object) -> float:
    """Get scalar item from tensor.

    Args:
        cls (type): The backend class.
        data (object): The input tensor data.

    Returns:
        float: The scalar value.
    """
    return generic_item(cp, data)
