"""Implementation of math signal operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.eager.core_math_ops.math_signal`)."""

from __future__ import annotations

import builtins  # noqa: F401
from typing import Callable, Optional, Protocol, Union, cast  # noqa: F401

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import EagerValue, global_eager_registry


@global_eager_registry.register("Correlate")
def _correlate(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager correlate operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "correlate", None)
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    (a, v) = (args[0], args[1])
    mode = kwargs.get("mode", "valid") if hasattr(kwargs, "get") else "valid"
    mod = _resolve_eager_mod(backend_module)
    return cast(EagerValue, mod.correlate(mod.asarray(a), mod.asarray(v), mode=mode))


@global_eager_registry.register("WindowHann")
def _np_windowhann(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np windowhann operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "windowhann", getattr(backend_module, "windowhann", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, _resolve_eager_mod(backend_module).hanning(*args, **kwargs))
