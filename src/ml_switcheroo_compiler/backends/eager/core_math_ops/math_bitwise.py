"""Implementation of math bitwise operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.eager.core_math_ops.math_bitwise`)."""

from __future__ import annotations

import builtins  # noqa: F401
import typing

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    EagerTensorProtocol,
    EagerValue,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol


@global_eager_registry.register("Signbit")
def _signbit(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager signbit operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "signbit", None)
    if func:
        return typing.cast("EagerValue", func(*args, **kwargs))
    x = typing.cast(EagerTensorProtocol, args[0])
    return x < 0


@global_eager_registry.register("Packbits")
def _np_packbits(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np packbits operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "packbits", getattr(backend_module, "packbits", None))
    if func is not None:
        return typing.cast("EagerValue", func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).packbits(*args, **kwargs)


@global_eager_registry.register("Unpackbits")
def _np_unpackbits(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np unpackbits operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "unpackbits", getattr(backend_module, "unpackbits", None))
    if func is not None:
        return typing.cast("EagerValue", func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).unpackbits(*args, **kwargs)
