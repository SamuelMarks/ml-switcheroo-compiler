# ruff: noqa: E402, E501, F403
"""Backend utilities."""

import numpy as np

from ml_switcheroo_compiler.backends.eager import generic_array


def zeros(cls: type, shape: tuple[int, ...]):
    """Create a tensor of zeros.

    Args:
        cls (type): The backend class.
        shape (tuple[int, ...]): The shape of the tensor.

    Returns: np.ndarray: The zeros tensor.
    """
    return np.zeros(shape)


def array(cls: type, data, dtype=None):
    """Create a tensor from data.

    Args:
        cls (type): The backend class.
        data (object): The input data.
        dtype (object): The target dtype.

    Returns: np.ndarray: The array tensor.
    """
    return generic_array(np, data, dtype)


def asarray(cls: type, data):
    """Convert data to an array.

    Args:
        cls (type): The backend class.
        data (object): The input data.

    Returns: np.ndarray: The array tensor.
    """
    return np.asarray(data)


def item(cls: type, data) -> float:
    """Get scalar item from tensor.

    Args:
        cls (type): The backend class.
        data (object): The input tensor data.

    Returns:
        float: The scalar value.
    """
    return np.asarray(data).item()
