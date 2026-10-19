"""Implementation of types operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.cupy.types`)."""

from typing import Optional, cast

from ml_switcheroo_compiler.backends.eager import (
    generic_array,
    generic_asarray,
    generic_item,
    generic_zeros,
)
from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerValue

try:
    import cupy as cp
except ImportError:
    cp = None


def zeros(cls: type, shape: tuple[int, ...]) -> EagerValue:
    """Create a tensor of zeros.

    Args:
        cls (type): The backend class.
        shape (tuple[int, ...]): The shape of the tensor.

    Returns:
        EagerValue: The zeros tensor.
    """
    return generic_zeros(cp, shape)


def array(cls: type, data: EagerValue, dtype: Optional[EagerValue] = None) -> EagerValue:
    """Create a tensor from data.

    Args:
        cls (type): The backend class.
        data: The input data.
        dtype: The target dtype.

    Returns:
        EagerValue: The array tensor.
    """
    return cast(EagerValue, cast("EagerCallableProtocol", generic_array)(cp, data, dtype))


def asarray(cls: type, data: EagerValue) -> EagerValue:
    """Convert data to an array.

    Args:
        cls (type): The backend class.
        data: The input data.

    Returns:
        EagerValue: The array tensor.
    """
    return cast(EagerValue, cast("EagerCallableProtocol", generic_asarray)(cp, data))


def item(cls: type, data: EagerValue) -> float:
    """Get scalar item from tensor.

    Args:
        cls (type): The backend class.
        data: The input tensor data.

    Returns:
        float: The scalar value.
    """
    return cast(float, cast("EagerCallableProtocol", generic_item)(cp, data))
