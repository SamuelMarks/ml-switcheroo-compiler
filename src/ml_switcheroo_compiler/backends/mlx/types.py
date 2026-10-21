"""Implementation of types operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.mlx.types`)."""

from __future__ import annotations

from types import ModuleType

from ml_switcheroo_compiler.backends.eager import (
    generic_array,
    generic_asarray,
    generic_item,
    generic_zeros,
)
from ml_switcheroo_compiler.backends.eager_registry import EagerValue, as_callable, as_float, as_tensor

mx: ModuleType | None
try:
    import mlx.core as _mx

    mx = _mx
except ImportError:
    mx = None


def zeros(cls: type, shape: tuple[int, ...]) -> EagerValue:
    """Execute zeros for compiler processing.

    Args:
        cls: Class.
        shape (tuple[int, ...]): Shape.

    Returns:
            EagerValue: Result.
    """
    return as_tensor(as_callable(generic_zeros)(mx, shape))


def array(cls: type, data: EagerValue, dtype: EagerValue | None = None) -> EagerValue:
    """Execute array for compiler processing.

    Args:
        cls: Class.
        data: Data.
        dtype: Dtype.

    Returns:
        EagerValue: Result.
    """
    del cls
    actual_dtype = getattr(mx, dtype) if isinstance(dtype, str) and hasattr(mx, dtype) else dtype
    return as_tensor(as_callable(generic_array)(mx, data, actual_dtype))


def asarray(cls: type, data: EagerValue) -> EagerValue:
    """Execute asarray for compiler processing.

    Args:
        cls: Class.
        data: Data.

    Returns:
            EagerValue: Result.
    """
    if hasattr(mx, "asarray"):
        return as_tensor(as_callable(generic_asarray)(mx, data))
    return as_tensor(as_callable(generic_array)(mx, data))


def item(cls: type, data: EagerValue) -> float:
    """Execute item for compiler processing.

    Args:
        cls: Class.
        data: Data.

    Returns:
        float: Result.
    """
    return as_float(as_callable(generic_item)(mx, data))
