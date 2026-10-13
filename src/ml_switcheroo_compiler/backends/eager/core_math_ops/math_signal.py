# ruff: noqa: E402, E501, F403
"""math_signal module."""

from __future__ import annotations

import builtins  # noqa: F401
from typing import Any, Callable, Optional, Protocol, Union  # noqa: F401

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry


@global_eager_registry.register("Correlate")
def _correlate(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
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
        return func(*args, **kwargs)
    import numpy as np

    (a, v) = (args[0], args[1])
    mode = kwargs.get("mode", "valid") if hasattr(kwargs, "get") else "valid"
    return np.correlate(np.asarray(a), np.asarray(v), mode=mode)


@global_eager_registry.register("WindowHann")
def _np_windowhann(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
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
        return func(*args, **kwargs)
    import numpy as np

    return np.hanning(*args, **kwargs)
