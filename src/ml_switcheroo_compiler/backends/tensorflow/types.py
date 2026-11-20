"""Operations and utilities for `ml_switcheroo_compiler.backends.tensorflow.types`."""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types

from ml_switcheroo_compiler.backends.eager import (
    generic_array,
    generic_asarray,
    generic_item,
    generic_zeros,
)
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol

_has_tf = sys.modules.get("tensorflow") is not None or (
    "tensorflow" not in sys.modules and importlib.util.find_spec("tensorflow") is not None
)
tf: types.ModuleType | BackendModuleProtocol | None = importlib.import_module("tensorflow") if _has_tf else None


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
    return generic_zeros(tf, shape)


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
    return generic_array(tf, data, dtype)


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
    return generic_asarray(tf, data)


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
    return generic_item(tf, data)
