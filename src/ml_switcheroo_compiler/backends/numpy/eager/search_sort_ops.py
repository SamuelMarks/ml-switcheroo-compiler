"""Numpy Search and Sort Ops."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, ensure_callable, ensure_eager_tensor, numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.reductions import _top_k
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, get_backend_member


@numpy_eager_registry.register("SortKeyVal")
def _np_sort_key_val(backend_module: BackendModuleProtocol, keys: EagerValue, values: EagerValue, axis: int = -1, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np sort key val operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        keys (EagerValue): Input `keys` argument (keys).
        values (EagerValue): Input `values` argument (values).
        axis (int): Input `axis` argument (axis).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    keys_arr = np.asarray(keys)
    values_arr = np.asarray(values)
    idx = np.argsort(keys_arr, axis=axis)
    sorted_keys = np.take_along_axis(keys_arr, idx, axis=axis)
    sorted_values = np.take_along_axis(values_arr, idx, axis=axis)
    return (sorted_keys, sorted_values)


@numpy_eager_registry.register("Partition")
def _np_partition(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Partition op.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return ensure_eager_tensor(get_backend_member(np, "partition")(get_backend_member(np, "asarray")(args[0]), args[1], *args[2:], **kwargs))


@numpy_eager_registry.register("Percentile")
def _np_percentile(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Percentile op.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return ensure_eager_tensor(get_backend_member(np, "percentile")(get_backend_member(np, "asarray")(args[0]), args[1], *args[2:], **kwargs))


@numpy_eager_registry.register("Quantile")
def _np_quantile(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Quantile op.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return ensure_eager_tensor(get_backend_member(np, "quantile")(get_backend_member(np, "asarray")(args[0]), args[1], *args[2:], **kwargs))


@numpy_eager_registry.register("Unique")
def _np_unique(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Compute unique op.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return ensure_eager_tensor(get_backend_member(np, "unique")(get_backend_member(np, "asarray")(args[0]), *args[1:], **kwargs))


@numpy_eager_registry.register("ArgSort")
def _np_argsort(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np argsort operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return backend_module.argsort(*args, **kwargs)


@numpy_eager_registry.register("Sort")
def _np_sort(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np sort operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    is_stable = kwargs.pop("is_stable", True)
    kwargs["kind"] = "stable" if is_stable else "quicksort"
    return backend_module.sort(*args, **kwargs)


@numpy_eager_registry.register("TopK")
def _np_top_k(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np top k operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return ensure_callable(_top_k)(*args, **kwargs)


@numpy_eager_registry.register("SearchSorted")
def _np_search_sorted(backend_module: BackendModuleProtocol, x: EagerValue, v: EagerValue, side: str = "left") -> EagerValue:
    """Execute the eager np search sorted operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        x (EagerValue): Input `x` argument (x).
        v (EagerValue): Input `v` argument (v).
        side (str): Input `side` argument (side).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return backend_module.searchsorted(x, v, side=side)


@numpy_eager_registry.register("Setdiff1d")
def _np_setdiff1d(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np setdiff1d operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return backend_module.setdiff1d(*args, **kwargs)


@numpy_eager_registry.register("Setxor1d")
def _np_setxor1d(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np setxor1d operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return backend_module.setxor1d(*args, **kwargs)


@numpy_eager_registry.register("SortComplex")
def _np_sort_complex(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np sort complex operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return backend_module.sort_complex(*args, **kwargs)
