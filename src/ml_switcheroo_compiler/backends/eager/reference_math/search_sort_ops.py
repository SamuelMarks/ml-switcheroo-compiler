"""Generic backend-parametric eager reference operations for `reference_math.search_sort_ops`."""

from __future__ import annotations

import types

from ml_switcheroo_compiler.backends.eager.reference_math.reductions import _top_k
from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module


@global_eager_registry.register("SortKeyVal")
def _np_sort_key_val(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    keys: BackendArg,
    values: BackendArg,
    axis: int = -1,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager np sort key val operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        keys (BackendArg): Input `keys` argument (keys).
        values (BackendArg): Input `values` argument (values).
        axis (int): Input `axis` argument (axis).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    keys_arr = arr_mod.asarray(keys)
    values_arr = arr_mod.asarray(values)
    idx = arr_mod.argsort(keys_arr, axis=axis)
    sorted_keys = arr_mod.take_along_axis(keys_arr, idx, axis=axis)
    sorted_values = arr_mod.take_along_axis(values_arr, idx, axis=axis)
    return (sorted_keys, sorted_values)


@global_eager_registry.register("Partition")
def _np_partition(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Partition op.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    return adapt_backend_module(arr_mod).require_symbol("partition")(
        adapt_backend_module(arr_mod).require_symbol("asarray")(args[0]), args[1], *args[2:], **kwargs
    )


@global_eager_registry.register("Percentile")
def _np_percentile(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Percentile op.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    return adapt_backend_module(arr_mod).require_symbol("percentile")(
        adapt_backend_module(arr_mod).require_symbol("asarray")(args[0]), args[1], *args[2:], **kwargs
    )


@global_eager_registry.register("Quantile")
def _np_quantile(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Quantile op.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    return adapt_backend_module(arr_mod).require_symbol("quantile")(
        adapt_backend_module(arr_mod).require_symbol("asarray")(args[0]), args[1], *args[2:], **kwargs
    )


@global_eager_registry.register("Unique")
def _np_unique(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Compute unique op.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    return adapt_backend_module(arr_mod).require_symbol("unique")(
        adapt_backend_module(arr_mod).require_symbol("asarray")(args[0]), *args[1:], **kwargs
    )


@global_eager_registry.register("ArgSort")
def _np_argsort(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np argsort operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("argsort")(*args, **kwargs)


@global_eager_registry.register("Sort")
def _np_sort(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np sort operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    is_stable = kwargs.pop("is_stable", True)
    kwargs["kind"] = "stable" if is_stable else "quicksort"
    return adapt_backend_module(backend_module).require_symbol("sort")(*args, **kwargs)


@global_eager_registry.register("TopK")
def _np_top_k(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np top k operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.backends.eager_registry import require_callable

    return require_callable(_top_k)(*args, **kwargs)


@global_eager_registry.register("SearchSorted")
def _np_search_sorted(
    backend_module: BackendModuleProtocol | types.ModuleType | None, x: BackendArg, v: BackendArg, side: str = "left"
) -> BackendArg:
    """Execute the eager np search sorted operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        x (BackendArg): Input `x` argument (x).
        v (BackendArg): Input `v` argument (v).
        side (str): Input `side` argument (side).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("searchsorted")(x, v, side=side)


@global_eager_registry.register("Setdiff1d")
def _np_setdiff1d(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np setdiff1d operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("setdiff1d")(*args, **kwargs)


@global_eager_registry.register("Setxor1d")
def _np_setxor1d(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np setxor1d operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("setxor1d")(*args, **kwargs)


@global_eager_registry.register("SortComplex")
def _np_sort_complex(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np sort complex operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("sort_complex")(*args, **kwargs)
