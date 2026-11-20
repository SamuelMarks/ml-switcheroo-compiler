"""Operations and utilities for `ml_switcheroo_compiler.backends.dask.types`."""

from __future__ import annotations

import typing
from types import ModuleType

from ml_switcheroo_compiler.backends.eager_registry import BackendArg, coerce_float
from ml_switcheroo_compiler.backends.protocols import adapt_backend_module

da: ModuleType | None
try:
    import dask.array as _da

    da = _da
except ImportError:
    da = None


@typing.runtime_checkable
class _ValueHolderProtocol(typing.Protocol):
    """Protocol for dtype objects exposing `.value`."""

    value: BackendArg


def zeros(cls: type | None, shape: tuple[int, ...]) -> BackendArg:
    """Create a tensor of zeros.

    Args:
        cls (type | None): The backend class.
        shape (tuple[int, ...]): The shape of the tensor.

    Returns:
        BackendArg: The zeros tensor.
    """
    assert da is not None
    return adapt_backend_module(da).require_op("zeros")(shape, chunks="auto")


def array(cls: type | None, data: BackendArg, dtype: BackendArg | None = None) -> BackendArg:
    """Create a tensor from data.

    Args:
        cls (type | None): The backend class.
        data (BackendArg): The input data.
        dtype (BackendArg | None): The target dtype.

    Returns:
        BackendArg: The array tensor.
    """
    assert da is not None
    if dtype is not None:
        raw_dt: BackendArg | _ValueHolderProtocol = dtype
        dt_val = raw_dt.value if isinstance(raw_dt, _ValueHolderProtocol) else dtype
        return adapt_backend_module(da).require_op("array")(data, dtype=dt_val)
    return adapt_backend_module(da).require_op("array")(data)


def asarray(cls: type | None, data: BackendArg) -> BackendArg:
    """Convert data to an array.

    Args:
        cls (type | None): The backend class.
        data (BackendArg): The input data.

    Returns:
        BackendArg: The array tensor.
    """
    assert da is not None
    return adapt_backend_module(da).require_op("asarray")(data)


def item(cls: type | None, data: BackendArg) -> float:
    """Get scalar item from tensor.

    Args:
        cls (type | None): The backend class.
        data (BackendArg): The input tensor data.

    Returns:
        float: The scalar value.
    """
    assert da is not None
    arr = da.asarray(data)
    comp = arr.compute()
    return float(coerce_float(comp.item()))
