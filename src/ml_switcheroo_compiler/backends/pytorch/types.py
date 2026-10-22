"""Implementation of types operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.pytorch.types`)."""

from __future__ import annotations

import importlib.util
import sys
import typing

if typing.TYPE_CHECKING:
    import torch
else:
    _has_torch = sys.modules.get("torch") is not None or ("torch" not in sys.modules and importlib.util.find_spec("torch") is not None)
    torch = importlib.import_module("torch") if _has_torch else None

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, as_tensor


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
    return as_tensor(torch.zeros(shape))


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
        dtype_str = str(getattr(dtype, "value", dtype)).split(".")[-1]
        dt = getattr(torch, dtype_str, None)
        return as_tensor(torch.tensor(data, dtype=dt))
    return as_tensor(torch.tensor(data))


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
    return as_tensor(torch.as_tensor(data))


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
