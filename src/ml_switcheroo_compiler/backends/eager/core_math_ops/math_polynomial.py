# ruff: noqa: E402, E501, F403
"""math_polynomial module."""

from __future__ import annotations

import builtins  # noqa: F401
from typing import Any

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry


@global_eager_registry.register("Polyval")
def _polyval(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _polyval operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    func = getattr(backend_module, "polyval", None)
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Polyint")
def _np_polyint(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_polyint operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    func = getattr(backend_module, "polyint", getattr(backend_module, "polyint", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return np.polyint(*args, **kwargs)
