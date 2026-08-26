# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Core utilities."""

from __future__ import annotations

from typing import Any

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry


@global_eager_registry.register("Signbit")
def _signbit(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _signbit operation.

    Args:
        backend_module (Any): The backend_module parameter.
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            Any: Result.
    """
    func = getattr(backend_module, "signbit", None)
    if func:
        return func(*args, **kwargs)
    x = args[0]
    return x < 0


@global_eager_registry.register("Packbits")
def _np_packbits(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_packbits operation.

    Args:
        backend_module (Any): The backend_module parameter.
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            Any: Result.
    """
    func = getattr(backend_module, "packbits", getattr(backend_module, "packbits", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return np.packbits(*args, **kwargs)


@global_eager_registry.register("Unpackbits")
def _np_unpackbits(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_unpackbits operation.

    Args:
        backend_module (Any): The backend_module parameter.
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            Any: Result.
    """
    func = getattr(backend_module, "unpackbits", getattr(backend_module, "unpackbits", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return np.unpackbits(*args, **kwargs)
