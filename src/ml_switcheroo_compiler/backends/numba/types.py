"""Numba backend type conversions and array creation utilities."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.eager_registry import EagerValue
from ml_switcheroo_compiler.backends.numba import kernels


def zeros(shape: tuple[int, ...]) -> EagerValue:
    """Create a tensor of zeros for Numba execution.

    Args:
        shape (tuple[int, ...]): The shape of the tensor.

    Returns:
        EagerValue: The allocated zeros array.
    """
    return kernels.zeros(shape, dtype=kernels.float32)


def array(
    data: EagerValue,
    dtype: str | None = None,
) -> EagerValue:
    """Create a tensor from data for Numba execution.

    Args:
        data (EagerValue): Input data sequence or buffer.
        dtype (str | None): The target dtype name.

    Returns:
        EagerValue: The created array buffer.
    """
    if dtype is not None:
        return kernels.array(data, dtype=getattr(dtype, "value", dtype))
    return kernels.array(data)


def asarray(
    data: EagerValue,
) -> EagerValue:
    """Convert input data to an array buffer for Numba execution.

    Args:
        data (EagerValue): Input data sequence or buffer.

    Returns:
        EagerValue: The array buffer representation.
    """
    return kernels.asarray(data)


def item(data: EagerValue) -> float:
    """Extract scalar item value from tensor.

    Args:
        data (EagerValue): Input tensor data.

    Returns:
        float: The extracted scalar float value.
    """
    return float(getattr(kernels.asarray(data), "item")())
