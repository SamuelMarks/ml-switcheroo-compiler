"""Operations and utilities for `ml_switcheroo_compiler.backends.cupy.types`."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.eager import (
    generic_array,
    generic_asarray,
    generic_item,
    generic_zeros,
)
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
)

try:
    import cupy as cp
except ImportError:
    cp = None


def zeros(cls: type | None, shape: tuple[int, ...]) -> BackendArg:
    """Create a tensor of zeros.

    Args:
        cls (type | None): The backend class.
        shape (tuple[int, ...]): The shape of the tensor.

    Returns:
        BackendArg: The zeros tensor.
    """
    return generic_zeros(cp, shape)


def array(cls: type | None, data: BackendArg, dtype: BackendArg | None = None) -> BackendArg:
    """Create a tensor from data.

    Args:
        cls (type | None): The backend class.
        data: The input data.
        dtype: The target dtype.

    Returns:
        BackendArg: The array tensor.
    """
    return generic_array(cp, data, dtype)


def asarray(cls: type | None, data: BackendArg) -> BackendArg:
    """Convert data to an array.

    Args:
        cls (type | None): The backend class.
        data: The input data.

    Returns:
        BackendArg: The array tensor.
    """
    return generic_asarray(cp, data)


def item(cls: type | None, data: BackendArg) -> float:
    """Get scalar item from tensor.

    Args:
        cls (type | None): The backend class.
        data: The input tensor data.

    Returns:
        float: The scalar value.
    """
    return generic_item(cp, data)
