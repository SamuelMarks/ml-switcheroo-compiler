"""Numpy Search and Sort Ops."""

from types import ModuleType
from typing import Callable, cast

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue, numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.reductions import _top_k


@numpy_eager_registry.register("SortKeyVal")
def _np_sort_key_val(backend_module: ModuleType, keys: EagerValue, values: EagerValue, axis: int = -1, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np sort key val operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        keys (EagerValue): The keys parameter.
        values (EagerValue): The values parameter.
        axis (int): The axis parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    keys_arr = np.asarray(keys)
    values_arr = np.asarray(values)
    idx = np.argsort(keys_arr, axis=axis)
    sorted_keys = np.take_along_axis(keys_arr, idx, axis=axis)
    sorted_values = np.take_along_axis(values_arr, idx, axis=axis)
    return (sorted_keys, sorted_values)


@numpy_eager_registry.register("Partition")
def _np_partition(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Partition op.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, getattr(np, "partition")(getattr(np, "asarray")(args[0]), args[1], *args[2:], **kwargs))


@numpy_eager_registry.register("Percentile")
def _np_percentile(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Percentile op.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, getattr(np, "percentile")(getattr(np, "asarray")(args[0]), args[1], *args[2:], **kwargs))


@numpy_eager_registry.register("Quantile")
def _np_quantile(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Quantile op.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, getattr(np, "quantile")(getattr(np, "asarray")(args[0]), args[1], *args[2:], **kwargs))


@numpy_eager_registry.register("Unique")
def _np_unique(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Compute unique op.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, getattr(np, "unique")(getattr(np, "asarray")(args[0]), *args[1:], **kwargs))


@numpy_eager_registry.register("ArgSort")
def _np_argsort(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np argsort operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, backend_module.argsort(*args, **kwargs))


@numpy_eager_registry.register("Sort")
def _np_sort(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np sort operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    is_stable = kwargs.pop("is_stable", True)
    kwargs["kind"] = "stable" if is_stable else "quicksort"
    return cast(EagerValue, backend_module.sort(*args, **kwargs))


@numpy_eager_registry.register("TopK")
def _np_top_k(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np top k operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, cast("Callable[..., EagerTensorProtocol]", _top_k)(*args, **kwargs))


@numpy_eager_registry.register("SearchSorted")
def _np_search_sorted(backend_module: ModuleType, x: EagerValue, v: EagerValue, side: str = "left") -> EagerValue:
    """Execute the eager np search sorted operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        x (EagerValue): The x parameter.
        v (EagerValue): The v parameter.
        side (str): The side parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, backend_module.searchsorted(x, v, side=side))


@numpy_eager_registry.register("Setdiff1d")
def _np_setdiff1d(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np setdiff1d operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, backend_module.setdiff1d(*args, **kwargs))


@numpy_eager_registry.register("Setxor1d")
def _np_setxor1d(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np setxor1d operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, backend_module.setxor1d(*args, **kwargs))


@numpy_eager_registry.register("SortComplex")
def _np_sort_complex(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np sort complex operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, backend_module.sort_complex(*args, **kwargs))
