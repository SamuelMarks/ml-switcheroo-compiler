"""Implementation of types operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.mlx.types`)."""

from __future__ import annotations

from types import ModuleType

from ml_switcheroo_compiler.backends.eager import (
    generic_array,
    generic_asarray,
    generic_item,
    generic_zeros,
)
from ml_switcheroo_compiler.backends.eager_registry import EagerValue, ensure_callable, ensure_eager_tensor, ensure_float
from ml_switcheroo_compiler.backends.protocols import get_backend_member, has_backend_member

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
    return ensure_eager_tensor(ensure_callable(generic_zeros)(mx, shape))


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
    actual_dtype = get_backend_member(mx, dtype) if isinstance(dtype, str) and has_backend_member(mx, dtype) else dtype
    return ensure_eager_tensor(ensure_callable(generic_array)(mx, data, actual_dtype))


def asarray(cls: type, data: EagerValue) -> EagerValue:
    """Execute asarray for compiler processing.

    Args:
        cls: Class.
        data: Data.

    Returns:
            EagerValue: Result.
    """
    if has_backend_member(mx, "asarray"):
        return ensure_eager_tensor(ensure_callable(generic_asarray)(mx, data))
    return ensure_eager_tensor(ensure_callable(generic_array)(mx, data))


def item(cls: type, data: EagerValue) -> float:
    """Execute item for compiler processing.

    Args:
        cls: Class.
        data: Data.

    Returns:
        float: Result.
    """
    return ensure_float(ensure_callable(generic_item)(mx, data))
