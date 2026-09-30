"""Bessel mathematical functions for numpy eager backend."""

from __future__ import annotations

from types import ModuleType
from typing import Union

import numpy as np
import scipy.special as sc

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_general import (
    _get_np_arg,
    _get_sc,
)


@numpy_eager_registry.register("BesselI0e")
def _np_bessel_i0e(
    backend_module: ModuleType,
    *args: Union[np.ndarray, float, int],
    **kwargs: Union[float, int, str, bool, None],
) -> np.ndarray:
    """Evaluate exponentially scaled modified Bessel function of order 0.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int]): Positional arguments.
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Computed exponentially scaled Bessel i0e result.
    """
    return backend_module.array(sc.i0e(args[0]))


@numpy_eager_registry.register("BesselI1e")
def _np_bessel_i1e(
    backend_module: ModuleType,
    *args: Union[np.ndarray, float, int],
    **kwargs: Union[float, int, str, bool, None],
) -> np.ndarray:
    """Evaluate exponentially scaled modified Bessel function of order 1.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int]): Positional arguments.
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Computed exponentially scaled Bessel i1e result.
    """
    return backend_module.array(sc.i1e(args[0]))


@numpy_eager_registry.register("modified_bessel_i0")
def _np_modified_bessel_i0(
    backend_module: ModuleType,
    *args: Union[np.ndarray, float, int],
    **kwargs: Union[float, int, str, bool, None],
) -> Union[np.ndarray, float, None]:
    """Evaluate modified Bessel function of order 0.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int]): Positional arguments.
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        Union[np.ndarray, float, None]: Result of modified Bessel i0 or None if input missing.
    """
    x = _get_np_arg(args, 0)
    if x is None:
        return None
    return np.i0(x)


@numpy_eager_registry.register("modified_bessel_i1")
def _np_modified_bessel_i1(
    backend_module: ModuleType,
    *args: Union[np.ndarray, float, int],
    **kwargs: Union[float, int, str, bool, None],
) -> Union[np.ndarray, float, None]:
    """Evaluate modified Bessel function of order 1.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int]): Positional arguments.
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        Union[np.ndarray, float, None]: Result of modified Bessel i1 or None if input missing.
    """
    sc_module = _get_sc()
    x = _get_np_arg(args, 0)
    if x is None:
        return None
    if sc_module is None:
        t = np.linspace(0.0, np.pi, 100)
        dt = float(t[1] - t[0])
        t_grid = np.reshape(t, (1,) * np.ndim(x) + (-1,)) if np.ndim(x) > 0 else t
        x_ex = np.expand_dims(x, -1) if np.ndim(x) > 0 else x
        integrand = np.exp(x_ex * np.cos(t_grid)) * np.cos(t_grid)
        trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
        return (1.0 / np.pi) * trapz_fn(integrand, dx=dt, axis=-1)
    return sc_module.i1(x)


@numpy_eager_registry.register("modified_bessel_k0")
def _np_modified_bessel_k0(
    backend_module: ModuleType,
    *args: Union[np.ndarray, float, int],
    **kwargs: Union[float, int, str, bool, None],
) -> Union[np.ndarray, float, None]:
    """Evaluate modified Bessel function of second kind of order 0.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int]): Positional arguments.
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        Union[np.ndarray, float, None]: Result of modified Bessel k0 or None if input missing.
    """
    sc_module = _get_sc()
    x = _get_np_arg(args, 0)
    if x is None:
        return None
    if sc_module is None:
        t = np.linspace(0.0, 10.0, 100)
        dt = float(t[1] - t[0])
        t_grid = np.reshape(t, (1,) * np.ndim(x) + (-1,)) if np.ndim(x) > 0 else t
        x_ex = np.expand_dims(x, -1) if np.ndim(x) > 0 else x
        integrand = np.exp(-x_ex * np.cosh(t_grid))
        trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
        return trapz_fn(integrand, dx=dt, axis=-1)
    return sc_module.k0(x)


@numpy_eager_registry.register("modified_bessel_k1")
def _np_modified_bessel_k1(
    backend_module: ModuleType,
    *args: Union[np.ndarray, float, int],
    **kwargs: Union[float, int, str, bool, None],
) -> Union[np.ndarray, float, None]:
    """Evaluate modified Bessel function of second kind of order 1.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int]): Positional arguments.
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        Union[np.ndarray, float, None]: Result of modified Bessel k1 or None if input missing.
    """
    sc_module = _get_sc()
    x = _get_np_arg(args, 0)
    if x is None:
        return None
    if sc_module is None:
        t = np.linspace(0.0, 10.0, 100)
        dt = float(t[1] - t[0])
        t_grid = np.reshape(t, (1,) * np.ndim(x) + (-1,)) if np.ndim(x) > 0 else t
        x_ex = np.expand_dims(x, -1) if np.ndim(x) > 0 else x
        integrand = np.exp(-x_ex * np.cosh(t_grid)) * np.cosh(t_grid)
        trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
        return trapz_fn(integrand, dx=dt, axis=-1)
    return sc_module.k1(x)
