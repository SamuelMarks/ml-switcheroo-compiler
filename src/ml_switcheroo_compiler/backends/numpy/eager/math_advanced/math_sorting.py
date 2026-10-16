"""math_sorting module."""

from __future__ import annotations

from types import ModuleType
from typing import Union, cast

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("ArgPartition")
def _np_argpartition(
    backend_module: ModuleType,
    *args: Union[np.ndarray[tuple[int, ...], np.dtype[np.generic]], int, tuple[int, ...], None],
    **kwargs: Union[int, tuple[int, ...], str, None],
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Perform an indirect partition along the given axis.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, int, tuple[int, ...], None]): Variable positional arguments.
        **kwargs (Union[int, tuple[int, ...], str, None]): Arbitrary keyword arguments.

    Returns:
        np.ndarray: The computed partition indices.
    """
    return cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], backend_module.argpartition(*args, **kwargs))


@numpy_eager_registry.register("Lexsort")
def _np_lexsort_(
    backend_module: ModuleType,
    *args: Union[np.ndarray[tuple[int, ...], np.dtype[np.generic]], tuple[np.ndarray[tuple[int, ...], np.dtype[np.generic]], ...], list[np.ndarray[tuple[int, ...], np.dtype[np.generic]]]],
    **kwargs: Union[int, str, None],
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Implement Lexsort via lexsort.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, tuple[np.ndarray, ...], list[np.ndarray]]): Variable positional arguments.
        **kwargs (Union[int, str, None]): Arbitrary keyword arguments.

    Returns:
        np.ndarray: The computed lexicographical sort order.
    """
    return cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], backend_module.lexsort(*args, **kwargs))


@numpy_eager_registry.register("Median")
def _np_median_(
    backend_module: ModuleType,
    *args: Union[np.ndarray[tuple[int, ...], np.dtype[np.generic]], int, tuple[int, ...], None],
    **kwargs: Union[int, tuple[int, ...], bool, None],
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Implement Median via median.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, int, tuple[int, ...], None]): Variable positional arguments.
        **kwargs (Union[int, tuple[int, ...], bool, None]): Arbitrary keyword arguments.

    Returns:
        np.ndarray: The computed median value.
    """
    return cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], backend_module.median(*args, **kwargs))
