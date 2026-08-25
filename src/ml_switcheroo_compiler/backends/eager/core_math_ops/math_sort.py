# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Core utilities."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry


@global_eager_registry.register("Argsort")
def _argsort(backend_module: object, a: object, axis: int = -1, **kwargs: object) -> object:
    """Evaluate _argsort operation.

    Args:
        backend_module (object): The backend_module parameter.
        a (object): The a parameter.
        axis (int): The axis parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.argsort(a, axis=axis)


@global_eager_registry.register("Lexsort")
def _lexsort(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _lexsort operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.lexsort(*args, **kwargs)


@global_eager_registry.register("Searchsorted")
def _searchsorted(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _searchsorted operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.searchsorted(*args, **kwargs)


@global_eager_registry.register("SortComplex")
def _sortcomplex(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _sortcomplex operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.sort_complex(*args, **kwargs)
