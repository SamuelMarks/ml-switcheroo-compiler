"""Operations and utilities for `ml_switcheroo_compiler.backends.numpy.types`."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.backends.eager import generic_array
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_float,
)


def zeros(cls: type | None, shape: tuple[int, ...]) -> BackendArg:
    """Create a tensor of zeros.

    Args:
        cls (type | None): The backend class.
        shape (tuple[int, ...]): The shape of the tensor.

    Returns: np.ndarray: The zeros tensor.
    """
    return np.zeros(shape)


def array(cls: type | None, data: BackendArg, dtype: BackendArg = None) -> BackendArg:
    """Create a tensor from data.

    Args:
        cls (type | None): The backend class.
        data (BackendArg): The input data.
        dtype (BackendArg): The target dtype.

    Returns: np.ndarray: The array tensor.
    """
    return generic_array(np, data, dtype)


def asarray(cls: type | None, data: BackendArg) -> BackendArg:
    """Convert data to an array.

    Args:
        cls (type | None): The backend class.
        data (BackendArg): The input data.

    Returns: np.ndarray: The array tensor.
    """
    return np.asarray(data)


def item(cls: type | None, data: BackendArg) -> float:
    """Get scalar item from tensor.

    Args:
        cls (type | None): The backend class.
        data (BackendArg): The input tensor data.

    Returns:
        float: The scalar value.
    """
    return coerce_float(np.asarray(data).item())
