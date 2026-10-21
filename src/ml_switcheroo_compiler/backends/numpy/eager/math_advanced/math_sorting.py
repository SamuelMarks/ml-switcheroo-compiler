"""Implementation of math sorting operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_sorting`)."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import as_ndarray
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol


@numpy_eager_registry.register("ArgPartition")
def _np_argpartition(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: int | tuple[int, ...] | str | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Perform an indirect partition along the given axis.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, int, tuple[int, ...], None]): Variable positional arguments.
        **kwargs (Union[int, tuple[int, ...], str, None]): Arbitrary keyword arguments.

    Returns:
        np.ndarray: The computed partition indices.
    """
    return as_ndarray(backend_module.argpartition(*args, **kwargs))


@numpy_eager_registry.register("Lexsort")
def _np_lexsort_(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: int | str | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Implement Lexsort via lexsort.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, tuple[np.ndarray, ...], list[np.ndarray]]): Variable positional arguments.
        **kwargs (Union[int, str, None]): Arbitrary keyword arguments.

    Returns:
        np.ndarray: The computed lexicographical sort order.
    """
    return as_ndarray(backend_module.lexsort(*args, **kwargs))


@numpy_eager_registry.register("Median")
def _np_median_(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: int | tuple[int, ...] | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Implement Median via median.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, int, tuple[int, ...], None]): Variable positional arguments.
        **kwargs (Union[int, tuple[int, ...], bool, None]): Arbitrary keyword arguments.

    Returns:
        np.ndarray: The computed median value.
    """
    return as_ndarray(backend_module.median(*args, **kwargs))
