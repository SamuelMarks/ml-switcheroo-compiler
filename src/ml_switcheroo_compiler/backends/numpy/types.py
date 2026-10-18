"""Backend utilities."""

from typing import Callable, cast

import numpy as np

from ml_switcheroo_compiler.backends.eager import generic_array
from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


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
    return cast(EagerValue, cast("Callable[..., EagerTensorProtocol]", generic_array)(np, data, dtype))


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
    return cast(float, np.asarray(data).item())
