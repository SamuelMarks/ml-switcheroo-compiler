"""Implementation of types operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.jax.types`)."""

from __future__ import annotations

import importlib.util
import sys
import typing

if typing.TYPE_CHECKING:
    import jax.numpy as jnp
else:
    _has_jax = sys.modules.get("jax") is not None or ("jax" not in sys.modules and importlib.util.find_spec("jax") is not None)
    jnp = importlib.import_module("jax.numpy") if _has_jax else None

from ml_switcheroo_compiler.backends.eager import (
    generic_array,
    generic_asarray,
    generic_item,
    generic_zeros,
)
from ml_switcheroo_compiler.backends.eager.types_utils import ZerosModule
from ml_switcheroo_compiler.backends.eager_registry import EagerValue, as_callable, as_float, as_tensor


def zeros(cls: type, shape: tuple[int, ...]) -> EagerValue:
    """Execute zeros for compiler processing.

    Args:
        cls: Class.
        shape (tuple[int, ...]): Shape.

    Returns:
            EagerValue: Result.
    """
    return generic_zeros(typing.cast(ZerosModule, jnp), shape)


def array(cls: type, data: EagerValue, dtype: EagerValue | None = None) -> EagerValue:
    """Execute array for compiler processing.

    Args:
        cls: Class.
        data: Data.
        dtype: Dtype.

    Returns:
            EagerValue: Result.
    """
    return as_tensor(as_callable(generic_array)(jnp, data, dtype))


def asarray(cls: type, data: EagerValue) -> EagerValue:
    """Execute asarray for compiler processing.

    Args:
        cls: Class.
        data: Data.

    Returns:
            EagerValue: Result.
    """
    return as_tensor(as_callable(generic_asarray)(jnp, data))


def item(cls: type, data: EagerValue) -> float:
    """Execute item for compiler processing.

    Args:
        cls: Class.
        data: Data.

    Returns:
        float: Result.
    """
    return as_float(as_callable(generic_item)(jnp, data))
