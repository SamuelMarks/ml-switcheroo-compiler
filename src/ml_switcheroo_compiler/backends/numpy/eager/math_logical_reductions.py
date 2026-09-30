"""Numpy Logical Reductions."""

from __future__ import annotations

from types import ModuleType
from typing import Union

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("All")
def _np_all(
    backend_module: ModuleType,
    *args: Union[np.ndarray, int, float, bool, None],
    **kwargs: Union[int, tuple[int, ...], bool, None],
) -> np.ndarray:
    """Evaluate _np_all operation.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, int, float, bool, None]): Positional arguments to pass to all.
        **kwargs (Union[int, tuple[int, ...], bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated boolean reduction.
    """
    return backend_module.all(*args, **kwargs)


@numpy_eager_registry.register("CountNonzero")
def _np_count_nonzero(
    backend_module: ModuleType,
    *args: Union[np.ndarray, int, float, bool, None],
    **kwargs: Union[int, tuple[int, ...], bool, None],
) -> np.ndarray:
    """Evaluate _np_count_nonzero operation.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, int, float, bool, None]): Positional arguments to pass to count_nonzero.
        **kwargs (Union[int, tuple[int, ...], bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Count of non-zero elements.
    """
    return backend_module.count_nonzero(*args, **kwargs)
