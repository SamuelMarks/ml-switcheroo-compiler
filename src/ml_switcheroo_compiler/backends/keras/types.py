# ruff: noqa: E402, E501, F403
"""Backend utilities."""

from typing import Optional

try:
    import keras.ops as kops
except Exception:
    kops = None

from ml_switcheroo_compiler.backends.eager import (
    generic_array,
    generic_asarray,
    generic_item,
    generic_zeros,
)


def zeros(cls: type, shape: tuple[int, ...]) -> object:
    """Create a tensor of zeros.

    Args:
        cls: The backend class.
        shape (tuple[int, ...]): The shape of the tensor.

    Returns:
        object: The zeros tensor.
    """
    return generic_zeros(kops, shape)


def array(cls: type, data: object, dtype: Optional[object] = None) -> object:
    """Create a tensor from data.

    Args:
        cls: The backend class.
        data: The input data.
        dtype: The target dtype.

    Returns:
        object: The array tensor.
    """
    return generic_array(kops, data, dtype)


def asarray(cls: type, data: object) -> object:
    """Convert data to an array.

    Args:
        cls: The backend class.
        data: The input data.

    Returns:
        object: The array tensor.
    """
    del cls
    if hasattr(kops, "asarray"):
        return generic_asarray(kops, data)
    return generic_array(kops, data)


def item(cls: type, data: object) -> float:
    """Get scalar item from tensor.

    Args:
        cls: The backend class.
        data: The input tensor data.

    Returns:
        float: The scalar value.
    """
    return generic_item(kops, data)
