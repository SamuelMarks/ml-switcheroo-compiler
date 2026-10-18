"""math_signal module."""

from __future__ import annotations

import builtins  # noqa: F401
from typing import Callable, Optional, Protocol, Union, cast  # noqa: F401

from ml_switcheroo_compiler.backends.eager.utils import host_backend
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
    return cast(EagerValue, host_backend.correlate(host_backend.asarray(a), host_backend.asarray(v), mode=mode))


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
    return cast(EagerValue, host_backend.hanning(*args, **kwargs))
