"""Implementation of math sort operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.eager.core_math_ops.math_sort`)."""

from __future__ import annotations

import builtins  # noqa: F401

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, ensure_eager_tensor, global_eager_registry
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, get_backend_member


@global_eager_registry.register("Argsort")
def _argsort(backend_module: BackendModuleProtocol, a: EagerValue, axis: int = -1, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager argsort operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        a: Input `a` argument (a).
        axis (int): Input `axis` argument (axis).
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    return ensure_eager_tensor(get_backend_member(backend_module, "argsort")(a, axis=axis))


@global_eager_registry.register("Lexsort")
def _lexsort(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager lexsort operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    return ensure_eager_tensor(get_backend_member(backend_module, "lexsort")(*args, **kwargs))


@global_eager_registry.register("Searchsorted")
def _searchsorted(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager searchsorted operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    return ensure_eager_tensor(get_backend_member(backend_module, "searchsorted")(*args, **kwargs))


@global_eager_registry.register("SortComplex")
def _sortcomplex(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager sortcomplex operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    return ensure_eager_tensor(get_backend_member(backend_module, "sort_complex")(*args, **kwargs))
