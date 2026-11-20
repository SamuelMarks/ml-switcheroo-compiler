"""Operations and utilities for `ml_switcheroo_compiler.backends.eager.core_math_ops.math_sorting`."""

from __future__ import annotations

import builtins  # noqa: F401
import types

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module


@global_eager_registry.register("Argpartition")
def _argpartition(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    a: BackendArg,
    kth: BackendArg,
    axis: int = -1,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager argpartition operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        a: Input `a` argument (a).
        kth: Input `kth` argument (kth).
        axis (int): Input `axis` argument (axis).
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return (
        adapt_backend_module(backend_module).require_symbol("argsort")(a, axis=axis)
        if adapt_backend_module(backend_module).has_symbol("argsort")
        else a
    )


@global_eager_registry.register("Median")
def _median(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager median operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return adapt_backend_module(backend_module).require_symbol("median")(*args, **kwargs)


@global_eager_registry.register("Percentile")
def _percentile(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager percentile operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return adapt_backend_module(backend_module).require_symbol("percentile")(*args, **kwargs)


@global_eager_registry.register("Quantile")
def _quantile(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager quantile operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return adapt_backend_module(backend_module).require_symbol("quantile")(*args, **kwargs)


@global_eager_registry.register("Partition")
def _np_partition(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np partition operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "partition", adapt_backend_module(backend_module).get_symbol("partition", None)
    )
    if func is not None:
        return func(*args, **kwargs)
    return _resolve_eager_mod(backend_module).partition(*args, **kwargs)


@global_eager_registry.register("Argsort")
def _argsort(
    backend_module: BackendModuleProtocol | types.ModuleType | None, a: BackendArg, axis: int = -1, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager argsort operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        a: Input `a` argument (a).
        axis (int): Input `axis` argument (axis).
        **kwargs: Keyword args.

    Returns:
        BackendArg: Computed `BackendArg` output.
    """
    del kwargs
    return adapt_backend_module(backend_module).require_symbol("argsort")(a, axis=axis)


@global_eager_registry.register("Lexsort")
def _lexsort(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager lexsort operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
        BackendArg: Computed `BackendArg` output.
    """
    return adapt_backend_module(backend_module).require_symbol("lexsort")(*args, **kwargs)


@global_eager_registry.register("Searchsorted")
def _searchsorted(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager searchsorted operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
        BackendArg: Computed `BackendArg` output.
    """
    return adapt_backend_module(backend_module).require_symbol("searchsorted")(*args, **kwargs)


@global_eager_registry.register("SortComplex")
def _sortcomplex(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager sortcomplex operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
        BackendArg: Computed `BackendArg` output.
    """
    return adapt_backend_module(backend_module).require_symbol("sort_complex")(*args, **kwargs)
