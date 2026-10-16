"""math_set module."""

from __future__ import annotations

from types import ModuleType
from typing import Union, cast

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("Union1d")
def _np_union1d(
    backend_module: ModuleType,
    *args: Union[np.ndarray[tuple[int, ...], np.dtype[np.generic]], list[int], list[float]],
    **kwargs: Union[bool, None],
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
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
    backend_module: ModuleType,
    *args: Union[np.ndarray[tuple[int, ...], np.dtype[np.generic]], list[int], list[float]],
    **kwargs: Union[bool, None],
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Implement Intersect1d via intersect1d.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, list[int], list[float]]): Variable positional arguments.
        **kwargs (Union[bool, None]): Arbitrary keyword arguments.

    Returns:
        np.ndarray: The computed intersection of two arrays.
    """
    return cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], backend_module.intersect1d(*args, **kwargs))


@numpy_eager_registry.register("Isin")
def _np_isin_(
    backend_module: ModuleType,
    *args: Union[np.ndarray[tuple[int, ...], np.dtype[np.generic]], list[int], list[float]],
    **kwargs: Union[bool, None],
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Implement Isin via isin.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, list[int], list[float]]): Variable positional arguments.
        **kwargs (Union[bool, None]): Arbitrary keyword arguments.

    Returns:
        np.ndarray: Boolean array of element containment.
    """
    return cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], backend_module.isin(*args, **kwargs))
