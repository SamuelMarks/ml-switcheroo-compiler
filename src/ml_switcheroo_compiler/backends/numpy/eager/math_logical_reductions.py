"""Numpy Logical Reductions."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import as_ndarray
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol


@numpy_eager_registry.register("All")
def _np_all(
    backend_module: BackendModuleProtocol,
    *args: np.ndarray[tuple[int, ...], np.dtype[np.float64]] | int | float | bool | None,
    **kwargs: int | tuple[int, ...] | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Execute the eager np all operation on the active backend array.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, int, float, bool, None]): Positional arguments to pass to all.
        **kwargs (Union[int, tuple[int, ...], bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated boolean reduction.
    """
    return as_ndarray(backend_module.all(*args, **kwargs))


@numpy_eager_registry.register("CountNonzero")
def _np_count_nonzero(
    backend_module: BackendModuleProtocol,
    *args: np.ndarray[tuple[int, ...], np.dtype[np.float64]] | int | float | bool | None,
    **kwargs: int | tuple[int, ...] | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Execute the eager np count nonzero operation on the active backend array.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, int, float, bool, None]): Positional arguments to pass to count_nonzero.
        **kwargs (Union[int, tuple[int, ...], bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Count of non-zero elements.
    """
    return as_ndarray(backend_module.count_nonzero(*args, **kwargs))
