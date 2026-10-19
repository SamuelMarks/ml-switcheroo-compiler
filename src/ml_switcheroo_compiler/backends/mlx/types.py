"""Implementation of types operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.mlx.types`)."""

from types import ModuleType
from typing import Optional, cast

from ml_switcheroo_compiler.backends.eager import (
    generic_array,
    generic_asarray,
    generic_item,
    generic_zeros,
)
from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerValue

mx: Optional[ModuleType]
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
    return cast(EagerValue, cast("EagerCallableProtocol", generic_zeros)(mx, shape))


def array(cls: type, data: EagerValue, dtype: Optional[EagerValue] = None) -> EagerValue:
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
    return cast(EagerValue, cast("EagerCallableProtocol", generic_array)(mx, data, actual_dtype))


def asarray(cls: type, data: EagerValue) -> EagerValue:
    """Execute asarray for compiler processing.

    Args:
        cls: Class.
        data: Data.

    Returns:
            EagerValue: Result.
    """
    if hasattr(mx, "asarray"):
        return cast(EagerValue, cast("EagerCallableProtocol", generic_asarray)(mx, data))
    return cast(EagerValue, cast("EagerCallableProtocol", generic_array)(mx, data))


def item(cls: type, data: EagerValue) -> float:
    """Execute item for compiler processing.

    Args:
        cls: Class.
        data: Data.

    Returns:
        float: Result.
    """
    return cast(float, cast("EagerCallableProtocol", generic_item)(mx, data))
