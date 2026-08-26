# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""math_random_ext module."""

from __future__ import annotations

from typing import Any

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry


@global_eager_registry.register("Pshuffle")
def _pshuffle(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _pshuffle operation.

    Args:
        backend_module (Any): The backend_module parameter.
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            Any: Result.
    """
    if hasattr(backend_module, "lax") and hasattr(backend_module.lax, "pshuffle"):
        return backend_module.lax.pshuffle(*args, **kwargs)
    return args[0] if args else None


@global_eager_registry.register("Gumbel")
def _np_gumbel(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_gumbel operation.

    Args:
        backend_module (Any): The backend_module parameter.
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            Any: Result.
    """
    func = getattr(backend_module, "gumbel", getattr(backend_module, "gumbel", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return np.random.gumbel(*args, **kwargs)


@global_eager_registry.register("RngUniform")
def _np_rnguniform(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_rnguniform operation.

    Args:
        backend_module (Any): The backend_module parameter.
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            Any: Result.
    """
    func = getattr(backend_module, "rnguniform", getattr(backend_module, "rnguniform", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return np.random.uniform(*args, **kwargs)


@global_eager_registry.register("Wald")
def _np_wald(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_wald operation.

    Args:
        backend_module (Any): The backend_module parameter.
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            Any: Result.
    """
    func = getattr(backend_module, "wald", getattr(backend_module, "wald", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return np.random.wald(*args, **kwargs)
