"""Numba backend type conversions and array creation utilities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def zeros(shape_or_cls: object, shape: tuple[int, ...] | None = None) -> np.ndarray:
    """Create a NumPy tensor of zeros for Numba execution.

    Args:
        shape_or_cls (object): Shape tuple or backend class context.
        shape (tuple[int, ...] | None): The shape of the tensor when called as classmethod.

    Returns:
        np.ndarray: The allocated zeros array.
    """
    actual_shape = shape_or_cls if shape is None and isinstance(shape_or_cls, tuple) else (shape or (0,))
    return np.zeros(actual_shape, dtype=np.float32)


def array(
    data_or_cls: object,
    data: Sequence[float] | None = None,
    dtype: str | None = None,
) -> np.ndarray:
    """Create a NumPy tensor from data for Numba execution.

    Args:
        data_or_cls (object): Data sequence or backend class context.
        data (Sequence[float] | None): Input data sequence when called as classmethod.
        dtype (str | None): The target dtype name.

    Returns:
        np.ndarray: The created NumPy array.
    """
    actual_data = data_or_cls if data is None else data
    if dtype is not None:
        return np.array(actual_data, dtype=getattr(dtype, "value", dtype))
    return np.array(actual_data)


def asarray(
    data_or_cls: object,
    data: Sequence[float] | None = None,
) -> np.ndarray:
    """Convert input data to a NumPy array for Numba execution.

    Args:
        data_or_cls (object): Input data sequence or backend class context.
        data (Sequence[float] | None): Input data sequence when called as classmethod.

    Returns:
        np.ndarray: The NumPy array representation.
    """
    actual_data = data_or_cls if data is None else data
    return np.asarray(actual_data)


def item(data_or_cls: object, data: np.ndarray | None = None) -> float:
    """Extract scalar item value from tensor.

    Args:
        data_or_cls (object): Target data or backend class context.
        data (np.ndarray | None): Input tensor data when called as classmethod.

    Returns:
        float: The extracted scalar float value.
    """
    actual_data = data_or_cls if data is None else data
    return float(np.asarray(actual_data).item())
