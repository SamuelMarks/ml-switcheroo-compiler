# ruff: noqa: E402, E501, F403
"""Core utilities."""

from __future__ import annotations

import builtins  # noqa: F401
from typing import Any

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry


@global_eager_registry.register("Argsort")
def _argsort(backend_module: Any, a: object, axis: int = -1, **kwargs: Any) -> Any:
    """Evaluate _argsort operation.

    Args:
        backend_module: The backend_module parameter.
        a: The a parameter.
        axis (int): The axis parameter.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return backend_module.argsort(a, axis=axis)


@global_eager_registry.register("Lexsort")
def _lexsort(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _lexsort operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return backend_module.lexsort(*args, **kwargs)


@global_eager_registry.register("Searchsorted")
def _searchsorted(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _searchsorted operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return backend_module.searchsorted(*args, **kwargs)


@global_eager_registry.register("SortComplex")
def _sortcomplex(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _sortcomplex operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return backend_module.sort_complex(*args, **kwargs)
