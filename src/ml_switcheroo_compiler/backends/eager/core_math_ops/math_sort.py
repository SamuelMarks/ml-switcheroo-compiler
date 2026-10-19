"""Implementation of math sort operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.eager.core_math_ops.math_sort`)."""

from __future__ import annotations

import builtins  # noqa: F401
from typing import cast

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, global_eager_registry


@global_eager_registry.register("Argsort")
def _argsort(backend_module: EagerValue, a: EagerValue, axis: int = -1, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager argsort operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        a: The a parameter.
        axis (int): The axis parameter.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return cast(EagerValue, getattr(backend_module, "argsort")(a, axis=axis))


@global_eager_registry.register("Lexsort")
def _lexsort(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager lexsort operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return cast(EagerValue, getattr(backend_module, "lexsort")(*args, **kwargs))


@global_eager_registry.register("Searchsorted")
def _searchsorted(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager searchsorted operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return cast(EagerValue, getattr(backend_module, "searchsorted")(*args, **kwargs))


@global_eager_registry.register("SortComplex")
def _sortcomplex(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager sortcomplex operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return cast(EagerValue, getattr(backend_module, "sort_complex")(*args, **kwargs))
