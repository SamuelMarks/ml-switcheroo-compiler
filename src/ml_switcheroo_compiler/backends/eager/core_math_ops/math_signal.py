# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""math_signal module."""

from __future__ import annotations

from typing import Any, Callable, Optional

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry


@global_eager_registry.register("Correlate")
def _correlate(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _correlate operation.

    Args:
        backend_module (Any): The backend_module parameter.
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            Any: Result.
    """
    func = getattr(backend_module, "correlate", None)
    if func:
        return func(*args, **kwargs)
    (a, v) = (args[0], args[1])
    mode = kwargs.get("mode", "valid")
    return backend_module.correlate(backend_module.asarray(a), backend_module.asarray(v), mode=mode)


@global_eager_registry.register("WindowHann")
def _np_windowhann(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_windowhann operation.

    Args:
        backend_module (Any): The backend_module parameter.
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            Any: Result.
    """
    func = getattr(backend_module, "windowhann", getattr(backend_module, "windowhann", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return np.hanning(*args, **kwargs)
