"""Core utilities."""

from __future__ import annotations

import builtins  # noqa: F401
from typing import cast

from ml_switcheroo_compiler.backends.eager_registry import (
    EagerTensorProtocol,
    EagerValue,
    global_eager_registry,
)
from ml_switcheroo_compiler.distributed.host_collectives import np


@global_eager_registry.register("Signbit")
def _signbit(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _signbit operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    func = getattr(backend_module, "signbit", None)
    if func:
        return cast(EagerValue, func(*args, **kwargs))
    x = cast(EagerTensorProtocol, args[0])
    return x < 0


@global_eager_registry.register("Packbits")
def _np_packbits(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_packbits operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    func = getattr(backend_module, "packbits", getattr(backend_module, "packbits", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, np.packbits(*args, **kwargs))


@global_eager_registry.register("Unpackbits")
def _np_unpackbits(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_unpackbits operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    func = getattr(backend_module, "unpackbits", getattr(backend_module, "unpackbits", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, np.unpackbits(*args, **kwargs))
