"""Implementation of types operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.dask.types`)."""

from __future__ import annotations

import typing
from types import ModuleType

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, ensure_eager_tensor, ensure_float
from ml_switcheroo_compiler.backends.protocols import ensure_mod

da: ModuleType | None
try:
    import dask.array as _da

    da = _da
except ImportError:
    da = None


@typing.runtime_checkable
class _ValueHolderProtocol(typing.Protocol):
    """Protocol for dtype objects exposing `.value`."""

    value: EagerValue


def zeros(cls: type, shape: tuple[int, ...]) -> EagerValue:
    """Create a tensor of zeros.

    Args:
        cls (type): The backend class.
        shape (tuple[int, ...]): The shape of the tensor.

    Returns:
        EagerValue: The zeros tensor.
    """
    return ensure_eager_tensor(ensure_mod(da).zeros(shape, chunks="auto"))


def array(cls: type, data: EagerValue, dtype: EagerValue | None = None) -> EagerValue:
    """Create a tensor from data.

    Args:
        cls (type): The backend class.
        data: The input data.
        dtype: The target dtype.

    Returns:
        EagerValue: The array tensor.
    """
    if dtype is not None:
        raw_dt: EagerValue = dtype
        dt_val = raw_dt.value if isinstance(raw_dt, _ValueHolderProtocol) else dtype
        return ensure_eager_tensor(ensure_mod(da).array(data, dtype=dt_val))
    return ensure_eager_tensor(ensure_mod(da).array(data))


def asarray(cls: type, data: EagerValue) -> EagerValue:
    """Convert data to an array.

    Args:
        cls (type): The backend class.
        data: The input data.

    Returns:
        EagerValue: The array tensor.
    """
    return ensure_eager_tensor(ensure_mod(da).asarray(data))


def item(cls: type, data: EagerValue) -> float:
    """Get scalar item from tensor.

    Args:
        cls (type): The backend class.
        data: The input tensor data.

    Returns:
        float: The scalar value.
    """
    arr = ensure_mod(da).asarray(data)
    comp = arr.compute()
    return float(ensure_float(comp.item()))
