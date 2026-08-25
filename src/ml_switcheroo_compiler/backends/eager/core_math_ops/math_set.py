# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""math_set module."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry


@global_eager_registry.register("Intersect1d")
def _intersect1d(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _intersect1d operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.intersect1d(*args, **kwargs)


@global_eager_registry.register("Unique")
def _unique(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _unique operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.unique(*args, **kwargs)


@global_eager_registry.register("Union1d")
def _np_union1d(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_union1d operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    func: object = getattr(backend_module, "union1d", getattr(backend_module, "union1d", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return np.union1d(*args, **kwargs)


@global_eager_registry.register("Unique")
def _np_unique(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_unique operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    func: object = getattr(backend_module, "unique", getattr(backend_module, "unique", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return np.unique(*args, **kwargs)


@global_eager_registry.register("UniqueAll")
def _np_uniqueall(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_uniqueall operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    func: object = getattr(backend_module, "uniqueall", getattr(backend_module, "uniqueall", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return np.unique(args[0], return_index=True, return_inverse=True, return_counts=True)


@global_eager_registry.register("UniqueCounts")
def _np_uniquecounts(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_uniquecounts operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    func: object = getattr(backend_module, "uniquecounts", getattr(backend_module, "uniquecounts", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return np.unique(args[0], return_counts=True)


@global_eager_registry.register("UniqueValues")
def _np_uniquevalues(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_uniquevalues operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    func: object = getattr(backend_module, "uniquevalues", getattr(backend_module, "uniquevalues", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return np.unique(args[0])
