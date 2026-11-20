"""Operations and utilities for `ml_switcheroo_compiler.backends.pytorch.types`."""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types

from ml_switcheroo_compiler.backends.eager_registry import BackendArg, coerce_float
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module

_has_torch = sys.modules.get("torch") is not None or (
    "torch" not in sys.modules and importlib.util.find_spec("torch") is not None
)
torch: types.ModuleType | BackendModuleProtocol | None = importlib.import_module("torch") if _has_torch else None


def zeros(cls: type | None, shape: tuple[int, ...]) -> BackendArg:
    """Execute zeros for compiler processing.

    Args:
        cls (type | None): Class.
        shape (tuple[int, ...]): Shape.

    Args:
        cls (type | None): Class.
        shape (tuple[int, ...]): Shape.

    Args:
        cls (type | None): The class.
    shape (tuple[int, ...]): Input `shape` argument (shape).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(torch).zeros(shape)


def array(cls: type | None, data: BackendArg, dtype: BackendArg = None) -> BackendArg:
    """Execute array for compiler processing.

    Args:
        cls (type | None): Class.
        data (BackendArg): Data.
        dtype (BackendArg): Dtype.

    Args:
        cls (type | None): Class.
        data (BackendArg): Data.
        dtype (BackendArg): Dtype.

    Args:
        cls (type | None): The class.
    data (BackendArg): Input `data` argument (data).
        dtype (BackendArg): Input `dtype` argument (dtype).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    tensor_fn = adapt_backend_module(torch).require_op("tensor")
    if dtype is not None:
        dtype_str = str(adapt_backend_module(dtype).get_symbol("value", dtype)).split(".")[-1]
        dt = adapt_backend_module(torch).get_symbol(dtype_str, None)
        return tensor_fn(data, dtype=dt)
    return tensor_fn(data)


def asarray(cls: type | None, data: BackendArg) -> BackendArg:
    """Execute asarray for compiler processing.

    Args:
        cls (type | None): Class.
        data (BackendArg): Data.

    Args:
        cls (type | None): Class.
        data (BackendArg): Data.

    Args:
        cls (type | None): The class.
    data (BackendArg): Input `data` argument (data).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(torch).require_op("as_tensor")(data)


def item(cls: type | None, data: BackendArg) -> float:
    """Execute item for compiler processing.

    Args:
        cls (type | None): Class.
        data (BackendArg): Data.

    Args:
        cls (type | None): Class.
        data (BackendArg): Data.

    Args:
        cls (type | None): The class.
    data (BackendArg): Input `data` argument (data).

    Returns:
        float: Computed `float` output.
    """
    return coerce_float(adapt_backend_module(torch).require_op("as_tensor")(data).item())
