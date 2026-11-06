"""Implementation of types operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.pytorch.types`)."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, ensure_eager_tensor
from ml_switcheroo_compiler.backends.protocols import ensure_mod, get_backend_member

try:
    import torch
except ImportError:
    torch = ensure_mod(None)


def zeros(cls: type, shape: tuple[int, ...]) -> EagerValue:
    """Execute zeros for compiler processing.

    Args:
        cls (type): Class.
        shape (tuple): Shape.

    Args:
        cls (type): Class.
        shape (tuple): Shape.

    Args:
        cls (type): The class.
    shape (tuple): The shape parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return ensure_eager_tensor(torch.zeros(shape))


def array(cls: type, data: EagerValue, dtype: EagerValue = None) -> EagerValue:
    """Execute array for compiler processing.

    Args:
        cls (type): Class.
        data (EagerValue): Data.
        dtype (EagerValue): Dtype.

    Args:
        cls (type): Class.
        data (EagerValue): Data.
        dtype (EagerValue): Dtype.

    Args:
        cls (type): The class.
    data (EagerValue): The data parameter.
        dtype (EagerValue): The dtype parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    if dtype is not None:
        dtype_str = str(get_backend_member(dtype, "value", dtype)).split(".")[-1]
        dt = get_backend_member(torch, dtype_str, None)
        return ensure_eager_tensor(torch.tensor(data, dtype=dt))
    return ensure_eager_tensor(torch.tensor(data))


def asarray(cls: type, data: EagerValue) -> EagerValue:
    """Execute asarray for compiler processing.

    Args:
        cls (type): Class.
        data (EagerValue): Data.

    Args:
        cls (type): Class.
        data (EagerValue): Data.

    Args:
        cls (type): The class.
    data (EagerValue): The data parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return ensure_eager_tensor(torch.as_tensor(data))


def item(cls: type, data: EagerValue) -> float:
    """Execute item for compiler processing.

    Args:
        cls (type): Class.
        data (EagerValue): Data.

    Args:
        cls (type): Class.
        data (EagerValue): Data.

    Args:
        cls (type): The class.
    data (EagerValue): The data parameter.

    Returns:
        float: Result.
    """
    return float(torch.as_tensor(data).item())
