"""math_set module."""

from __future__ import annotations

from types import ModuleType
from typing import Union

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("Union1d")
def _np_union1d(
    backend_module: ModuleType,
    *args: Union[np.ndarray, list[int], list[float]],
    **kwargs: Union[bool, None],
) -> np.ndarray:
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
    *args: Union[np.ndarray, list[int], list[float]],
    **kwargs: Union[bool, None],
) -> np.ndarray:
    """Implement Intersect1d via intersect1d.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, list[int], list[float]]): Variable positional arguments.
        **kwargs (Union[bool, None]): Arbitrary keyword arguments.

    Returns:
        np.ndarray: The computed intersection of two arrays.
    """
    return backend_module.intersect1d(*args, **kwargs)


@numpy_eager_registry.register("Isin")
def _np_isin_(
    backend_module: ModuleType,
    *args: Union[np.ndarray, list[int], list[float]],
    **kwargs: Union[bool, None],
) -> np.ndarray:
    """Implement Isin via isin.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, list[int], list[float]]): Variable positional arguments.
        **kwargs (Union[bool, None]): Arbitrary keyword arguments.

    Returns:
        np.ndarray: Boolean array of element containment.
    """
    return backend_module.isin(*args, **kwargs)
