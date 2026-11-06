"""Implementation of types operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.numpy.types`)."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.backends.eager import generic_array
from ml_switcheroo_compiler.backends.eager_registry import EagerValue, ensure_callable, ensure_eager_tensor, ensure_float


def zeros(cls: type, shape: tuple[int, ...]) -> EagerValue:
    """Create a tensor of zeros.

    Args:
        cls (type): The backend class.
        shape (tuple[int, ...]): The shape of the tensor.

    Returns: np.ndarray: The zeros tensor.
    """
    return np.zeros(shape)


def array(cls: type, data: EagerValue, dtype: EagerValue = None) -> EagerValue:
    """Create a tensor from data.

    Args:
        cls (type): The backend class.
        data (EagerValue): The input data.
        dtype (EagerValue): The target dtype.

    Returns: np.ndarray: The array tensor.
    """
    return ensure_eager_tensor(ensure_callable(generic_array)(np, data, dtype))


def asarray(cls: type, data: EagerValue) -> EagerValue:
    """Convert data to an array.

    Args:
        cls (type): The backend class.
        data (EagerValue): The input data.

    Returns: np.ndarray: The array tensor.
    """
    return np.asarray(data)


def item(cls: type, data: EagerValue) -> float:
    """Get scalar item from tensor.

    Args:
        cls (type): The backend class.
        data (EagerValue): The input tensor data.

    Returns:
        float: The scalar value.
    """
    return ensure_float(np.asarray(data).item())
