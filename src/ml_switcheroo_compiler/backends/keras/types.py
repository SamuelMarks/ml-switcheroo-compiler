# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Backend utilities."""

from typing import Optional

import keras.ops as kops

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
    return generic_asarray(kops, data)


def item(cls: type, data: object) -> float:
    """Get scalar item from tensor.

    Args:
        cls: The backend class.
        data: The input tensor data.

    Returns:
        float: The scalar value.
    """
    return generic_item(kops, data)
