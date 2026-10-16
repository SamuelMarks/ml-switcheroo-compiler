"""math_set module."""

from __future__ import annotations

import builtins  # noqa: F401
from typing import cast

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, global_eager_registry
from ml_switcheroo_compiler.distributed.host_collectives import np


@global_eager_registry.register("Intersect1d")
def _intersect1d(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _intersect1d operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return cast(EagerValue, getattr(backend_module, "intersect1d")(*args, **kwargs))


@global_eager_registry.register("Union1d")
def _np_union1d(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
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
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, np.union1d(*args, **kwargs))


@global_eager_registry.register("Unique")
def _np_unique(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
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
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, np.unique(*args, **kwargs))


@global_eager_registry.register("UniqueAll")
def _np_uniqueall(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
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
        return cast(EagerValue, func(*args, **kwargs))
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
def _np_uniquecounts(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
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
        return cast(EagerValue, func(*args, **kwargs))
    arr = np.asarray(args[0]).flatten()
    values = np.unique(arr)
    counts_map: dict[EagerValue, int] = {}
    for elem in arr.tolist():
        counts_map[elem] = counts_map.get(elem, 0) + 1
    counts = np.array([counts_map[v] for v in values.tolist()], dtype=np.intp)
    return values, counts


@global_eager_registry.register("UniqueValues")
def _np_uniquevalues(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
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
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, np.unique(args[0]))
