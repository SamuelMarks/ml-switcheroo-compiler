"""Numba backend type conversions and array creation utilities."""

from collections.abc import Sequence
from typing import Optional

import numpy as np


def zeros(cls: type, shape: tuple[int, ...]) -> np.ndarray:
    """Create a NumPy tensor of zeros for Numba execution.

    Args:
        cls (type): The backend class context.
        shape (tuple[int, ...]): The shape of the tensor.

    Returns:
        np.ndarray: The allocated zeros array.
    """
    del cls
    return np.zeros(shape, dtype=np.float32)


def array(cls: type, data: Sequence[float], dtype: Optional[str] = None) -> np.ndarray:
    """Create a NumPy tensor from data for Numba execution.

    Args:
        cls (type): The backend class context.
        data (Sequence[float]): The input data sequence.
        dtype (Optional[str]): The target dtype name.

    Returns:
        np.ndarray: The created NumPy array.
    """
    del cls
    if dtype is not None:
        return np.array(data, dtype=getattr(dtype, "value", dtype))
    return np.array(data)


def asarray(cls: type, data: Sequence[float]) -> np.ndarray:
    """Convert input data to a NumPy array for Numba execution.

    Args:
        cls (type): The backend class context.
        data (Sequence[float]): The input data sequence.

    Returns:
        np.ndarray: The NumPy array representation.
    """
    del cls
    return np.asarray(data)


def item(cls: type, data: np.ndarray) -> float:
    """Extract scalar item value from tensor.

    Args:
        cls (type): The backend class context.
        data (np.ndarray): The input tensor data.

    Returns:
        float: The extracted scalar float value.
    """
    del cls
    return float(np.asarray(data).item())
