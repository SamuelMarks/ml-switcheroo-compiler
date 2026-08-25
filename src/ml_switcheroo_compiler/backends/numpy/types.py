# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Backend utilities."""

import numpy as np

from ml_switcheroo_compiler.backends.eager import generic_array


def zeros(cls: type, shape: tuple[int, ...]) -> object:
    """Create a tensor of zeros.

    Args:
        cls (type): The backend class.
        shape (tuple[int, ...]): The shape of the tensor.

    Returns: object: The zeros tensor.
    """
    return np.zeros(shape)


def array(cls: type, data: object, dtype: object = None) -> object:
    """Create a tensor from data.

    Args:
        cls (type): The backend class.
        data (object): The input data.
        dtype (object): The target dtype.

    Returns: object: The array tensor.
    """
    return generic_array(np, data, dtype)


def asarray(cls: type, data: object) -> object:
    """Convert data to an array.

    Args:
        cls (type): The backend class.
        data (object): The input data.

    Returns: object: The array tensor.
    """
    return np.asarray(data)


def item(cls: type, data: object) -> float:
    """Get scalar item from tensor.

    Args:
        cls (type): The backend class.
        data (object): The input tensor data.

    Returns:
        float: The scalar value.
    """
    return np.asarray(data).item()
