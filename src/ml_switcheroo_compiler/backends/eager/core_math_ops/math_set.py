"""math_set module."""

from __future__ import annotations

import builtins  # noqa: F401
from typing import cast

from ml_switcheroo_compiler.backends.eager.utils import host_backend
from ml_switcheroo_compiler.backends.eager_registry import EagerValue, global_eager_registry


@global_eager_registry.register("Intersect1d")
def _intersect1d(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager intersect1d operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return cast(EagerValue, getattr(backend_module, "intersect1d")(*args, **kwargs))


@global_eager_registry.register("Union1d")
def _np_union1d(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np union1d operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "union1d", getattr(backend_module, "union1d", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, host_backend.union1d(*args, **kwargs))


@global_eager_registry.register("Unique")
def _np_unique(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np unique operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "unique", getattr(backend_module, "unique", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, host_backend.unique(*args, **kwargs))


@global_eager_registry.register("UniqueAll")
def _np_uniqueall(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np uniqueall operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "uniqueall", getattr(backend_module, "uniqueall", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    arr = host_backend.asarray(args[0]).flatten()
    values, indices = host_backend.unique(arr, return_index=True)
    val_to_idx = {v: i for i, v in enumerate(values.tolist())}
    inverse_indices = host_backend.array([val_to_idx[x] for x in arr.tolist()], dtype=host_backend.intp)
    counts_dict: dict[int, int] = {}
    for idx in inverse_indices.tolist():
        counts_dict[idx] = counts_dict.get(idx, 0) + 1
    counts = host_backend.array([counts_dict[i] for i in range(len(values))], dtype=host_backend.intp)
    return values, indices, inverse_indices, counts


@global_eager_registry.register("UniqueCounts")
def _np_uniquecounts(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np uniquecounts operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "uniquecounts", getattr(backend_module, "uniquecounts", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    arr = host_backend.asarray(args[0]).flatten()
    values = host_backend.unique(arr)
    counts_map: dict[EagerValue, int] = {}
    for elem in arr.tolist():
        counts_map[elem] = counts_map.get(elem, 0) + 1
    counts = host_backend.array([counts_map[v] for v in values.tolist()], dtype=host_backend.intp)
    return values, counts


@global_eager_registry.register("UniqueValues")
def _np_uniquevalues(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np uniquevalues operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "uniquevalues", getattr(backend_module, "uniquevalues", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, host_backend.unique(args[0]))
