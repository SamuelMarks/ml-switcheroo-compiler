"""math_signal module."""

from __future__ import annotations

import builtins  # noqa: F401
from typing import Callable, Optional, Protocol, Union, cast  # noqa: F401

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, global_eager_registry
from ml_switcheroo_compiler.distributed.host_collectives import np


@global_eager_registry.register("Correlate")
def _correlate(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _correlate operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    func = getattr(backend_module, "correlate", None)
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    (a, v) = (args[0], args[1])
    mode = kwargs.get("mode", "valid") if hasattr(kwargs, "get") else "valid"
    return cast(EagerValue, np.correlate(np.asarray(a), np.asarray(v), mode=mode))


@global_eager_registry.register("WindowHann")
def _np_windowhann(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_windowhann operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    func = getattr(backend_module, "windowhann", getattr(backend_module, "windowhann", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, np.hanning(*args, **kwargs))
