"""Implementation of types operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.jax.types`)."""

from __future__ import annotations

import importlib.util
import sys
import typing

from ml_switcheroo_compiler.backends.eager import (
    generic_array,
    generic_asarray,
    generic_item,
    generic_zeros,
)
from ml_switcheroo_compiler.backends.eager_registry import EagerValue, ensure_callable, ensure_eager_tensor, ensure_float
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, ensure_mod


class _JaxTypesModProtocol(typing.Protocol):
    """Protocol for binding `jnp` onto `types` module."""

    jnp: BackendModuleProtocol


def _is_jax_types_mod(mod: EagerValue) -> typing.TypeGuard[_JaxTypesModProtocol]:
    """Type guard narrowing `mod` to `_JaxTypesModProtocol`.

    Args:
        mod (EagerValue): Module instance.

    Returns:
        bool: True if non-None.
    """
    return mod is not None


if typing.TYPE_CHECKING:
    import jax.numpy as jnp

_has_jax = sys.modules.get("jax") is not None or ("jax" not in sys.modules and importlib.util.find_spec("jax") is not None)
_jt_mod: EagerValue = sys.modules[__name__]
assert _is_jax_types_mod(_jt_mod)
_jt_mod.jnp = ensure_mod(importlib.import_module("jax.numpy") if _has_jax else None)


def zeros(cls: type, shape: tuple[int, ...]) -> EagerValue:
    """Execute zeros for compiler processing.

    Args:
        cls: Class.
        shape (tuple[int, ...]): Shape.

    Returns:
            EagerValue: Result.
    """
    return ensure_eager_tensor(ensure_callable(generic_zeros)(jnp, shape))


def array(cls: type, data: EagerValue, dtype: EagerValue | None = None) -> EagerValue:
    """Execute array for compiler processing.

    Args:
        cls: Class.
        data: Data.
        dtype: Dtype.

    Returns:
            EagerValue: Result.
    """
    return ensure_eager_tensor(ensure_callable(generic_array)(jnp, data, dtype))


def asarray(cls: type, data: EagerValue) -> EagerValue:
    """Execute asarray for compiler processing.

    Args:
        cls: Class.
        data: Data.

    Returns:
            EagerValue: Result.
    """
    return ensure_eager_tensor(ensure_callable(generic_asarray)(jnp, data))


def item(cls: type, data: EagerValue) -> float:
    """Execute item for compiler processing.

    Args:
        cls: Class.
        data: Data.

    Returns:
        float: Result.
    """
    return ensure_float(ensure_callable(generic_item)(jnp, data))
