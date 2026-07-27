# ruff: noqa: E501
"""Numpy Search and Sort Ops."""

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.reductions import _top_k


@numpy_eager_registry.register("SortKeyVal")
def _np_sort_key_val(backend_module: object, keys: object, values: object, axis: int = -1, *args: object, **kwargs: object) -> object:
    """Evaluate the sort key val logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        keys (object): Required parameter for keys.
        values (object): Required parameter for values.
        axis (int): Required parameter for axis.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    keys_arr = np.asarray(keys)
    values_arr = np.asarray(values)
    idx = np.argsort(keys_arr, axis=axis)
    sorted_keys = np.take_along_axis(keys_arr, idx, axis=axis)
    sorted_values = np.take_along_axis(values_arr, idx, axis=axis)
    return (sorted_keys, sorted_values)


@numpy_eager_registry.register("Partition")
def _np_partition(backend_module: object, *args: object, **kwargs: object) -> object:
    """Partition op."""
    return np.partition(np.asarray(args[0]), args[1], *args[2:], **kwargs)


@numpy_eager_registry.register("Percentile")
def _np_percentile(backend_module: object, *args: object, **kwargs: object) -> object:
    """Percentile op."""
    return np.percentile(np.asarray(args[0]), args[1], *args[2:], **kwargs)


@numpy_eager_registry.register("Quantile")
def _np_quantile(backend_module: object, *args: object, **kwargs: object) -> object:
    """Quantile op."""
    return np.quantile(np.asarray(args[0]), args[1], *args[2:], **kwargs)


@numpy_eager_registry.register("Unique")
def _np_unique(backend_module: object, *args: object, **kwargs: object) -> object:
    """Unique op."""
    return np.unique(np.asarray(args[0]), *args[1:], **kwargs)


@numpy_eager_registry.register("ArgSort")
def _np_argsort(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the argsort logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.argsort(*args, **kwargs)


@numpy_eager_registry.register("Sort")
def _np_sort(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the sort logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    is_stable = kwargs.pop("is_stable", True)
    kwargs.pop("dimension", None)
    kwargs.pop("dim", None)
    kwargs["kind"] = "stable" if is_stable else "quicksort"
    return backend_module.sort(*args, **kwargs)


@numpy_eager_registry.register("TopK")
def _np_top_k(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the top k logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return _top_k(*args, **kwargs)


@numpy_eager_registry.register("SearchSorted")
def _np_search_sorted(backend_module: object, x: object, v: object, side: str = "left") -> object:
    """Evaluate the search sorted logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        x (object): Required parameter for x.
        v (object): Required parameter for v.
        side (str): Required parameter for side.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.searchsorted(x, v, side=side)


@numpy_eager_registry.register("Setdiff1d")
def _np_setdiff1d(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the setdiff1d logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.setdiff1d(*args, **kwargs)


@numpy_eager_registry.register("Setxor1d")
def _np_setxor1d(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the setxor1d logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.setxor1d(*args, **kwargs)


@numpy_eager_registry.register("SortComplex")
def _np_sort_complex(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the sort complex logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.sort_complex(*args, **kwargs)
