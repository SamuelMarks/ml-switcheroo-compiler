"""Implementation of math set operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.eager.core_math_ops.math_set`)."""

from __future__ import annotations

import builtins  # noqa: F401
import typing

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import EagerValue, global_eager_registry
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol


@global_eager_registry.register("Intersect1d")
def _intersect1d(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager intersect1d operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return typing.cast("EagerValue", getattr(backend_module, "intersect1d")(*args, **kwargs))


@global_eager_registry.register("Union1d")
def _np_union1d(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
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
        return typing.cast("EagerValue", func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).union1d(*args, **kwargs)


@global_eager_registry.register("Unique")
def _np_unique(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
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
        return typing.cast("EagerValue", func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).unique(*args, **kwargs)


@global_eager_registry.register("UniqueAll")
def _np_uniqueall(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
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
        return typing.cast("EagerValue", func(*args, **kwargs))
    mod = _resolve_eager_mod(backend_module)
    arr = mod.asarray(args[0]).flatten()
    values, indices = mod.unique(arr, return_index=True)
    val_to_idx = {v: i for i, v in enumerate(values.tolist())}
    inverse_indices = mod.array([val_to_idx[x] for x in arr.tolist()], dtype=mod.intp)
    counts_dict: dict[int, int] = {}
    for idx in inverse_indices.tolist():
        counts_dict[idx] = counts_dict.get(idx, 0) + 1
    counts = mod.array([counts_dict[i] for i in range(len(values))], dtype=mod.intp)
    return values, indices, inverse_indices, counts


@global_eager_registry.register("UniqueCounts")
def _np_uniquecounts(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
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
        return typing.cast("EagerValue", func(*args, **kwargs))
    mod = _resolve_eager_mod(backend_module)
    arr = mod.asarray(args[0]).flatten()
    values = mod.unique(arr)
    counts_map: dict[EagerValue, int] = {}
    for elem in arr.tolist():
        counts_map[elem] = counts_map.get(elem, 0) + 1
    counts = mod.array([counts_map[v] for v in values.tolist()], dtype=mod.intp)
    return values, counts


@global_eager_registry.register("UniqueValues")
def _np_uniquevalues(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
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
        return typing.cast("EagerValue", func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).unique(args[0])
