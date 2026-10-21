"""Implementation of types operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.keras.types`)."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, as_callable, as_float, as_tensor

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


def zeros(cls: type, shape: tuple[int, ...]) -> EagerValue:
    """Create a tensor of zeros.

    Args:
        cls: The backend class.
        shape (tuple[int, ...]): The shape of the tensor.

    Returns:
        EagerValue: The zeros tensor.
    """
    return generic_zeros(kops, shape)


def array(cls: type, data: EagerValue, dtype: EagerValue | None = None) -> EagerValue:
    """Create a tensor from data.

    Args:
        cls: The backend class.
        data: The input data.
        dtype: The target dtype.

    Returns:
        EagerValue: The array tensor.
    """
    return as_tensor(as_callable(generic_array)(kops, data, dtype))


def asarray(cls: type, data: EagerValue) -> EagerValue:
    """Convert data to an array.

    Args:
        cls: The backend class.
        data: The input data.

    Returns:
        EagerValue: The array tensor.
    """
    del cls
    if hasattr(kops, "asarray"):
        return as_tensor(as_callable(generic_asarray)(kops, data))
    return as_tensor(as_callable(generic_array)(kops, data))


def item(cls: type, data: EagerValue) -> float:
    """Get scalar item from tensor.

    Args:
        cls: The backend class.
        data: The input tensor data.

    Returns:
        float: The scalar value.
    """
    return as_float(as_callable(generic_item)(kops, data))
