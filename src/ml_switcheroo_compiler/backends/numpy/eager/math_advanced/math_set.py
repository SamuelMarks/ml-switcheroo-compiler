"""Implementation of math set operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_set`)."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import as_ndarray
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol


@numpy_eager_registry.register("Union1d")
def _np_union1d(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Find the union of two one-dimensional arrays.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, list[int], list[float]]): Variable positional arguments.
        **kwargs (Union[bool, None]): Arbitrary keyword arguments.

    Returns:
        np.ndarray: The computed unique union.
    """
    return np.union1d(np.asarray(args[0]), np.asarray(args[1]), **kwargs)


@numpy_eager_registry.register("Intersect1d")
def _np_intersect1d_(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Implement Intersect1d via intersect1d.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, list[int], list[float]]): Variable positional arguments.
        **kwargs (Union[bool, None]): Arbitrary keyword arguments.

    Returns:
        np.ndarray: The computed intersection of two arrays.
    """
    return as_ndarray(backend_module.intersect1d(*args, **kwargs))


@numpy_eager_registry.register("Isin")
def _np_isin_(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Implement Isin via isin.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, list[int], list[float]]): Variable positional arguments.
        **kwargs (Union[bool, None]): Arbitrary keyword arguments.

    Returns:
        np.ndarray: Boolean array of element containment.
    """
    return as_ndarray(backend_module.isin(*args, **kwargs))
