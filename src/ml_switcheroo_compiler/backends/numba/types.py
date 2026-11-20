"""Numba backend type conversions and array creation utilities."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.eager_registry import BackendArg, coerce_float
from ml_switcheroo_compiler.backends.numba import kernels


@typing.runtime_checkable
class _ValueHolderProtocol(typing.Protocol):
    """Protocol for dtype objects exposing `.value`."""

    value: BackendArg


def zeros(shape: tuple[int, ...]) -> BackendArg:
    """Create a tensor of zeros for Numba execution.

    Args:
        shape (tuple[int, ...]): The shape of the tensor.

    Returns:
        BackendArg: The allocated zeros array.
    """
    return kernels.zeros(shape, dtype=kernels.float32)


def array(
    data: BackendArg,
    dtype: str | BackendArg | None = None,
) -> BackendArg:
    """Create a tensor from data for Numba execution.

    Args:
        data (BackendArg): Input data sequence or buffer.
        dtype (str | BackendArg | None): The target dtype name.

    Returns:
        BackendArg: The created array buffer.
    """
    if dtype is not None:
        raw_dt: BackendArg | _ValueHolderProtocol = dtype
        dt_val = raw_dt.value if isinstance(raw_dt, _ValueHolderProtocol) else dtype
        return kernels.array(data, dtype=dt_val)
    return kernels.array(data)


def asarray(
    data: BackendArg,
) -> BackendArg:
    """Convert input data to an array buffer for Numba execution.

    Args:
        data (BackendArg): Input data sequence or buffer.

    Returns:
        BackendArg: The array buffer representation.
    """
    return kernels.asarray(data)


def item(data: BackendArg) -> float:
    """Extract scalar item value from tensor.

    Args:
        data (BackendArg): Input tensor data.

    Returns:
        float: The extracted scalar float value.
    """
    from ml_switcheroo_compiler.backends.protocols import adapt_backend_module

    return float(coerce_float(adapt_backend_module(kernels.asarray(data)).require_op("item")()))
