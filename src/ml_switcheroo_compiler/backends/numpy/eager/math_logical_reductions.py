"""Numpy Logical Reductions."""

from __future__ import annotations

from types import ModuleType
from typing import Union, cast

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("All")
def _np_all(
    backend_module: ModuleType,
    *args: Union[np.ndarray[tuple[int, ...], np.dtype[np.generic]], int, float, bool, None],
    **kwargs: Union[int, tuple[int, ...], bool, None],
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Execute the eager np all operation on the active backend array.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, int, float, bool, None]): Positional arguments to pass to all.
        **kwargs (Union[int, tuple[int, ...], bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated boolean reduction.
    """
    return cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], backend_module.all(*args, **kwargs))


@numpy_eager_registry.register("CountNonzero")
def _np_count_nonzero(
    backend_module: ModuleType,
    *args: Union[np.ndarray[tuple[int, ...], np.dtype[np.generic]], int, float, bool, None],
    **kwargs: Union[int, tuple[int, ...], bool, None],
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Execute the eager np count nonzero operation on the active backend array.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, int, float, bool, None]): Positional arguments to pass to count_nonzero.
        **kwargs (Union[int, tuple[int, ...], bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Count of non-zero elements.
    """
    return cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], backend_module.count_nonzero(*args, **kwargs))
