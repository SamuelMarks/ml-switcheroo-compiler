"""Operations and utilities for `ml_switcheroo_compiler.backends.eager.core_math_ops.math_set`."""

from __future__ import annotations

import builtins  # noqa: F401
import types

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module


@global_eager_registry.register("Intersect1d")
def _intersect1d(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager intersect1d operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return adapt_backend_module(backend_module).require_symbol("intersect1d")(*args, **kwargs)


@global_eager_registry.register("Union1d")
def _np_union1d(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np union1d operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "union1d", adapt_backend_module(backend_module).get_symbol("union1d", None)
    )
    if func is not None:
        return func(*args, **kwargs)
    return _resolve_eager_mod(backend_module).union1d(*args, **kwargs)


@global_eager_registry.register("Unique")
def _np_unique(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np unique operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "unique", adapt_backend_module(backend_module).get_symbol("unique", None)
    )
    if func is not None:
        return func(*args, **kwargs)
    return _resolve_eager_mod(backend_module).unique(*args, **kwargs)


@global_eager_registry.register("UniqueAll")
def _np_uniqueall(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np uniqueall operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "uniqueall", adapt_backend_module(backend_module).get_symbol("uniqueall", None)
    )
    if func is not None:
        return func(*args, **kwargs)
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
def _np_uniquecounts(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np uniquecounts operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "uniquecounts", adapt_backend_module(backend_module).get_symbol("uniquecounts", None)
    )
    if func is not None:
        return func(*args, **kwargs)
    mod = _resolve_eager_mod(backend_module)
    arr = mod.asarray(args[0]).flatten()
    values = mod.unique(arr)
    counts_map: dict[BackendArg, int] = {}
    for elem in arr.tolist():
        counts_map[elem] = counts_map.get(elem, 0) + 1
    counts = mod.array([counts_map[v] for v in values.tolist()], dtype=mod.intp)
    return values, counts


@global_eager_registry.register("UniqueValues")
def _np_uniquevalues(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np uniquevalues operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "uniquevalues", adapt_backend_module(backend_module).get_symbol("uniquevalues", None)
    )
    if func is not None:
        return func(*args, **kwargs)
    return _resolve_eager_mod(backend_module).unique(args[0])
