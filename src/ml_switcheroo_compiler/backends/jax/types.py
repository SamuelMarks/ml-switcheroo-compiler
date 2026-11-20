"""Operations and utilities for `ml_switcheroo_compiler.backends.jax.types`."""

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

_has_jax = sys.modules.get("jax") is not None or (
    "jax" not in sys.modules and importlib.util.find_spec("jax") is not None
)
jnp: types.ModuleType | BackendModuleProtocol | None = importlib.import_module("jax.numpy") if _has_jax else None


def zeros(cls: type | None, shape: tuple[int, ...]) -> BackendArg:
    """Execute zeros for compiler processing.

    Args:
        cls: Class.
        shape (tuple[int, ...]): Shape.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return generic_zeros(jnp, shape)


def array(cls: type | None, data: BackendArg, dtype: BackendArg | None = None) -> BackendArg:
    """Execute array for compiler processing.

    Args:
        cls: Class.
        data: Data.
        dtype: Dtype.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return generic_array(jnp, data, dtype)


def asarray(cls: type | None, data: BackendArg) -> BackendArg:
    """Execute asarray for compiler processing.

    Args:
        cls: Class.
        data: Data.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return generic_asarray(jnp, data)


def item(cls: type | None, data: BackendArg) -> float:
    """Execute item for compiler processing.

    Args:
        cls: Class.
        data: Data.

    Returns:
        float: Computed `float` output.
    """
    return generic_item(jnp, data)
