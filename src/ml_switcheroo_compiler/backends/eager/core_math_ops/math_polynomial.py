# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""math_polynomial module."""

from __future__ import annotations

import builtins
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
