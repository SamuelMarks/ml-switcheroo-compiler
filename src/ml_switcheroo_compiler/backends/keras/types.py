"""Operations and utilities for `ml_switcheroo_compiler.backends.keras.types`."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
)
from ml_switcheroo_compiler.backends.protocols import adapt_backend_module

try:
    import keras.ops as kops
except (ImportError, ModuleNotFoundError, OSError, AttributeError, RuntimeError, TypeError, ValueError):
    kops = None

from ml_switcheroo_compiler.backends.eager import (
    generic_array,
    generic_asarray,
    generic_item,
    generic_zeros,
)


def zeros(cls: type | None, shape: tuple[int, ...]) -> BackendArg:
    """Create a tensor of zeros.

    Args:
        cls: The backend class.
        shape (tuple[int, ...]): The shape of the tensor.

    Returns:
        BackendArg: The zeros tensor.
    """
    return generic_zeros(kops, shape)


def array(cls: type | None, data: BackendArg, dtype: BackendArg | None = None) -> BackendArg:
    """Create a tensor from data.

    Args:
        cls: The backend class.
        data: The input data.
        dtype: The target dtype.

    Returns:
        BackendArg: The array tensor.
    """
    return generic_array(kops, data, dtype)


def asarray(cls: type | None, data: BackendArg) -> BackendArg:
    """Convert data to an array.

    Args:
        cls: The backend class.
        data: The input data.

    Returns:
        BackendArg: The array tensor.
    """
    del cls
    if adapt_backend_module(kops).resolve_symbol("asarray")[0]:
        return generic_asarray(kops, data)
    return generic_array(kops, data)


def item(cls: type | None, data: BackendArg) -> float:
    """Get scalar item from tensor.

    Args:
        cls: The backend class.
        data: The input tensor data.

    Returns:
        float: The scalar value.
    """
    del cls
    has_np, np_fn = adapt_backend_module(data).resolve_symbol("numpy")
    has_item = adapt_backend_module(data).resolve_symbol("item")[0]
    target = np_fn() if (has_np and np_fn is not None and not has_item) else data
    return generic_item(kops, target)
