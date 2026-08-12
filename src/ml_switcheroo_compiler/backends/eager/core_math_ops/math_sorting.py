# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""math_sorting module."""

from __future__ import annotations

from typing import Any

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry


@global_eager_registry.register("Argpartition")
def _argpartition(backend_module: Any, a: Any, kth: Any, axis: int = -1, **kwargs: Any) -> Any:
    """Evaluate _argpartition operation.

    Args:
        backend_module (object): The backend_module parameter.
        a (object): The a parameter.
        kth (object): The kth parameter.
        axis (int): The axis parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.argsort(a, axis=axis) if hasattr(backend_module, "argsort") else a


@global_eager_registry.register("Median")
def _median(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _median operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.median(*args, **kwargs)


@global_eager_registry.register("Percentile")
def _percentile(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _percentile operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.percentile(*args, **kwargs)


@global_eager_registry.register("Quantile")
def _quantile(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _quantile operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.quantile(*args, **kwargs)


@global_eager_registry.register("Partition")
def _np_partition(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_partition operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "partition", getattr(backend_module, "partition", None))
    if func is not None:
        try:
            return func(*args, **kwargs)
        except Exception:
            pass
    import numpy as np

    return np.partition(*args, **kwargs)
