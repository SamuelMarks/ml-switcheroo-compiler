# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""math_set module."""

from __future__ import annotations

import builtins
from typing import Any

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry


@global_eager_registry.register("Intersect1d")
def _intersect1d(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _intersect1d operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return backend_module.intersect1d(*args, **kwargs)


@global_eager_registry.register("Union1d")
def _np_union1d(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_union1d operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    func = getattr(backend_module, "union1d", getattr(backend_module, "union1d", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return np.union1d(*args, **kwargs)


@global_eager_registry.register("Unique")
def _np_unique(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_unique operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    func = getattr(backend_module, "unique", getattr(backend_module, "unique", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return np.unique(*args, **kwargs)


@global_eager_registry.register("UniqueAll")
def _np_uniqueall(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_uniqueall operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    func = getattr(backend_module, "uniqueall", getattr(backend_module, "uniqueall", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    arr = np.asarray(args[0]).flatten()
    values, indices = np.unique(arr, return_index=True)
    val_to_idx = {v: i for i, v in enumerate(values.tolist())}
    inverse_indices = np.array([val_to_idx[x] for x in arr.tolist()], dtype=np.intp)
    counts_dict: dict[int, int] = {}
    for idx in inverse_indices.tolist():
        counts_dict[idx] = counts_dict.get(idx, 0) + 1
    counts = np.array([counts_dict[i] for i in range(len(values))], dtype=np.intp)
    return values, indices, inverse_indices, counts


@global_eager_registry.register("UniqueCounts")
def _np_uniquecounts(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_uniquecounts operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    func = getattr(backend_module, "uniquecounts", getattr(backend_module, "uniquecounts", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    arr = np.asarray(args[0]).flatten()
    values = np.unique(arr)
    counts_map: dict[Any, int] = {}
    for elem in arr.tolist():
        counts_map[elem] = counts_map.get(elem, 0) + 1
    counts = np.array([counts_map[v] for v in values.tolist()], dtype=np.intp)
    return values, counts


@global_eager_registry.register("UniqueValues")
def _np_uniquevalues(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_uniquevalues operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    func = getattr(backend_module, "uniquevalues", getattr(backend_module, "uniquevalues", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return np.unique(args[0])
