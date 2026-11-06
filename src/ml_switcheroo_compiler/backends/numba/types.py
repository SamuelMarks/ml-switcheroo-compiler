"""Numba backend type conversions and array creation utilities."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, ensure_eager_tensor, ensure_float
from ml_switcheroo_compiler.backends.numba import kernels


@typing.runtime_checkable
class _ValueHolderProtocol(typing.Protocol):
    """Protocol for dtype objects exposing `.value`."""

    value: EagerValue


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
    dtype: str | EagerValue | None = None,
) -> EagerValue:
    """Create a tensor from data for Numba execution.

    Args:
        data (EagerValue): Input data sequence or buffer.
        dtype (str | None): The target dtype name.

    Returns:
        EagerValue: The created array buffer.
    """
    if dtype is not None:
        raw_dt: EagerValue = dtype
        dt_val = raw_dt.value if isinstance(raw_dt, _ValueHolderProtocol) else dtype
        return kernels.array(data, dtype=dt_val)
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
    return float(ensure_float(ensure_eager_tensor(kernels.asarray(data)).item()))
