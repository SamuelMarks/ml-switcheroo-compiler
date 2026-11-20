"""Operations and utilities for `ml_switcheroo_compiler.backends.mlx.types`."""

from __future__ import annotations

from types import ModuleType

from ml_switcheroo_compiler.backends.eager import (
    generic_array,
    generic_asarray,
    generic_item,
    generic_zeros,
)
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
)
from ml_switcheroo_compiler.backends.protocols import adapt_backend_module

mx: ModuleType | None
try:
    import mlx.core as _mx

    mx = _mx
except ImportError:
    mx = None


def zeros(cls: type | None, shape: tuple[int, ...]) -> BackendArg:
    """Execute zeros for compiler processing.

    Args:
        cls: Class.
        shape (tuple[int, ...]): Shape.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return generic_zeros(mx, shape)


def array(cls: type | None, data: BackendArg, dtype: BackendArg | None = None) -> BackendArg:
    """Execute array for compiler processing.

    Args:
        cls: Class.
        data: Data.
        dtype: Dtype.

    Returns:
        BackendArg: Computed `BackendArg` output.
    """
    del cls
    actual_dtype = (
        adapt_backend_module(mx).require_symbol(dtype)
        if isinstance(dtype, str) and adapt_backend_module(mx).has_symbol(dtype)
        else dtype
    )
    return generic_array(mx, data, actual_dtype)


def asarray(cls: type | None, data: BackendArg) -> BackendArg:
    """Execute asarray for compiler processing.

    Args:
        cls: Class.
        data: Data.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    if adapt_backend_module(mx).has_symbol("asarray"):
        return generic_asarray(mx, data)
    return generic_array(mx, data)


def item(cls: type | None, data: BackendArg) -> float:
    """Execute item for compiler processing.

    Args:
        cls: Class.
        data: Data.

    Returns:
        float: Computed `float` output.
    """
    return generic_item(mx, data)
