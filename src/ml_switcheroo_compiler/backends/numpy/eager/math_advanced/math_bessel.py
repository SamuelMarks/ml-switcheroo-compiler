"""Bessel mathematical functions for numpy eager backend."""

from __future__ import annotations

import importlib

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, as_callable, numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import as_ndarray
from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_general import (
    _get_np_arg,
    _get_sc,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol


@numpy_eager_registry.register("BesselI0e")
def _np_bessel_i0e(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Evaluate exponentially scaled modified Bessel function of order 0.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int]): Positional arguments.
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Computed exponentially scaled Bessel i0e result.
    """
    return as_ndarray(backend_module.array(importlib.import_module("scipy.special").i0e(args[0])))


@numpy_eager_registry.register("BesselI1e")
def _np_bessel_i1e(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Evaluate exponentially scaled modified Bessel function of order 1.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int]): Positional arguments.
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Computed exponentially scaled Bessel i1e result.
    """
    return as_ndarray(backend_module.array(importlib.import_module("scipy.special").i1e(args[0])))


@numpy_eager_registry.register("modified_bessel_i0")
def _np_modified_bessel_i0(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]] | float | None:
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
    return as_ndarray(getattr(np, "i0")(x))


@numpy_eager_registry.register("modified_bessel_i1")
def _np_modified_bessel_i1(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]] | float | None:
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
        trapz_fn = as_callable(getattr(np, "trapezoid", getattr(np, "trapz", None)))
        return as_ndarray(trapz_fn(integrand, dx=dt, axis=-1) * (1.0 / np.pi))
    return as_ndarray(getattr(sc_module, "i1")(x))


@numpy_eager_registry.register("modified_bessel_k0")
def _np_modified_bessel_k0(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]] | float | None:
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
        trapz_fn = as_callable(getattr(np, "trapezoid", getattr(np, "trapz", None)))
        return as_ndarray(trapz_fn(integrand, dx=dt, axis=-1))
    return as_ndarray(getattr(sc_module, "k0")(x))


@numpy_eager_registry.register("modified_bessel_k1")
def _np_modified_bessel_k1(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]] | float | None:
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
        trapz_fn = as_callable(getattr(np, "trapezoid", getattr(np, "trapz", None)))
        return as_ndarray(trapz_fn(integrand, dx=dt, axis=-1))
    return as_ndarray(getattr(sc_module, "k1")(x))
