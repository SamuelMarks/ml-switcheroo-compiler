"""Implementation of types operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.pytorch.types`)."""

from __future__ import annotations

import importlib.util
import sys
import typing

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, ensure_eager_tensor
from ml_switcheroo_compiler.backends.protocols import ensure_mod, get_backend_member

if typing.TYPE_CHECKING:
    import torch

_has_torch = sys.modules.get("torch") is not None or ("torch" not in sys.modules and importlib.util.find_spec("torch") is not None)
ensure_mod(sys.modules[__name__]).torch = ensure_mod(importlib.import_module("torch") if _has_torch else None)


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
    shape (tuple): Input `shape` argument (shape).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
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
    data (EagerValue): Input `data` argument (data).
        dtype (EagerValue): Input `dtype` argument (dtype).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
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
    data (EagerValue): Input `data` argument (data).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
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
    data (EagerValue): Input `data` argument (data).

    Returns:
        float: Computed `float` output.
    """
    return float(torch.as_tensor(data).item())
