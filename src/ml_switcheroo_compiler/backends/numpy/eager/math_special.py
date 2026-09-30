"""Numpy eager special math functions."""

from __future__ import annotations

from types import ModuleType
from typing import Union

import numpy as np
import scipy.special as sc

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("BesselJ0")
def _np_bessel_j0(
    backend_module: ModuleType,
    *args: Union[np.ndarray, float, int],
    **kwargs: Union[float, int, str, bool, None],
) -> np.ndarray:
    """Evaluate Bessel function of the first kind of order 0.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int]): Positional arguments (x).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated Bessel function array.
    """
    x = args[0] if args else kwargs.get("x", 0.0)
    return backend_module.array(sc.j0(x))


@numpy_eager_registry.register("BesselJ1")
def _np_bessel_j1(
    backend_module: ModuleType,
    *args: Union[np.ndarray, float, int],
    **kwargs: Union[float, int, str, bool, None],
) -> np.ndarray:
    """Evaluate Bessel function of the first kind of order 1.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int]): Positional arguments (x).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated Bessel function array.
    """
    x = args[0] if args else kwargs.get("x", 0.0)
    return backend_module.array(sc.j1(x))


@numpy_eager_registry.register("BesselK0")
def _np_bessel_k0(
    backend_module: ModuleType,
    *args: Union[np.ndarray, float, int],
    **kwargs: Union[float, int, str, bool, None],
) -> np.ndarray:
    """Evaluate modified Bessel function of the second kind of order 0.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int]): Positional arguments (x).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated modified Bessel function array.
    """
    x = args[0] if args else kwargs.get("x", 0.0)
    return backend_module.array(sc.k0(x))


@numpy_eager_registry.register("BesselK0e")
def _np_bessel_k0e(
    backend_module: ModuleType,
    *args: Union[np.ndarray, float, int],
    **kwargs: Union[float, int, str, bool, None],
) -> np.ndarray:
    """Evaluate exponentially scaled modified Bessel function K0e.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int]): Positional arguments (x).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated exponentially scaled Bessel array.
    """
    x = args[0] if args else kwargs.get("x", 0.0)
    return backend_module.array(sc.k0e(x))


@numpy_eager_registry.register("BesselK1")
def _np_bessel_k1(
    backend_module: ModuleType,
    *args: Union[np.ndarray, float, int],
    **kwargs: Union[float, int, str, bool, None],
) -> np.ndarray:
    """Evaluate modified Bessel function of the second kind of order 1.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int]): Positional arguments (x).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated modified Bessel function array.
    """
    x = args[0] if args else kwargs.get("x", 0.0)
    return backend_module.array(sc.k1(x))


@numpy_eager_registry.register("BesselK1e")
def _np_bessel_k1e(
    backend_module: ModuleType,
    *args: Union[np.ndarray, float, int],
    **kwargs: Union[float, int, str, bool, None],
) -> np.ndarray:
    """Evaluate exponentially scaled modified Bessel function K1e.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int]): Positional arguments (x).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated exponentially scaled Bessel array.
    """
    x = args[0] if args else kwargs.get("x", 0.0)
    return backend_module.array(sc.k1e(x))


@numpy_eager_registry.register("BesselY0")
def _np_bessel_y0(
    backend_module: ModuleType,
    *args: Union[np.ndarray, float, int],
    **kwargs: Union[float, int, str, bool, None],
) -> np.ndarray:
    """Evaluate Bessel function of the second kind of order 0.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int]): Positional arguments (x).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated Bessel Y0 array.
    """
    x = args[0] if args else kwargs.get("x", 0.0)
    return backend_module.array(sc.y0(x))


@numpy_eager_registry.register("BesselY1")
def _np_bessel_y1(
    backend_module: ModuleType,
    *args: Union[np.ndarray, float, int],
    **kwargs: Union[float, int, str, bool, None],
) -> np.ndarray:
    """Evaluate Bessel function of the second kind of order 1.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int]): Positional arguments (x).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated Bessel Y1 array.
    """
    x = args[0] if args else kwargs.get("x", 0.0)
    return backend_module.array(sc.y1(x))


@numpy_eager_registry.register("Dawsn")
def _np_dawsn(
    backend_module: ModuleType,
    *args: Union[np.ndarray, float, int],
    **kwargs: Union[float, int, str, bool, None],
) -> np.ndarray:
    """Evaluate Dawson's integral.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int]): Positional arguments (x).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated Dawson's integral array.
    """
    x = args[0] if args else kwargs.get("x", 0.0)
    return backend_module.array(sc.dawsn(x))


@numpy_eager_registry.register("Expint")
def _np_expint(
    backend_module: ModuleType,
    *args: Union[np.ndarray, float, int],
    **kwargs: Union[float, int, str, bool, None],
) -> np.ndarray:
    """Evaluate exponential integral E_n(x).

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int]): Positional arguments (x, n).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated exponential integral array.
    """
    x = args[0] if args else kwargs.get("x", 0.0)
    n = kwargs.get("n", 1)
    if len(args) > 1:
        n = args[1]
    return backend_module.array(sc.expn(n, x))


@numpy_eager_registry.register("FresnelCos")
def _np_fresnel_cos(
    backend_module: ModuleType,
    *args: Union[np.ndarray, float, int],
    **kwargs: Union[float, int, str, bool, None],
) -> np.ndarray:
    """Evaluate Fresnel cosine integral.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int]): Positional arguments (x).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated Fresnel cosine integral array.
    """
    x = args[0] if args else kwargs.get("x", 0.0)
    return backend_module.array(sc.fresnel(x)[1])


@numpy_eager_registry.register("FresnelSin")
def _np_fresnel_sin(
    backend_module: ModuleType,
    *args: Union[np.ndarray, float, int],
    **kwargs: Union[float, int, str, bool, None],
) -> np.ndarray:
    """Evaluate Fresnel sine integral.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int]): Positional arguments (x).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated Fresnel sine integral array.
    """
    x = args[0] if args else kwargs.get("x", 0.0)
    return backend_module.array(sc.fresnel(x)[0])


@numpy_eager_registry.register("Spence")
def _np_spence(
    backend_module: ModuleType,
    *args: Union[np.ndarray, float, int],
    **kwargs: Union[float, int, str, bool, None],
) -> np.ndarray:
    """Evaluate Spence's function (dilogarithm).

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int]): Positional arguments (x).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated Spence's function array.
    """
    x = args[0] if args else kwargs.get("x", 0.0)
    return backend_module.array(sc.spence(x))


@numpy_eager_registry.register("BesselI0")
def _np_bessel_i0(
    backend_module: ModuleType,
    x: Union[np.ndarray, float, int],
    **kwargs: Union[float, int, str, bool, None],
) -> np.ndarray:
    """Evaluate modified Bessel function of the first kind of order 0.

    Args:
        backend_module (ModuleType): Active backend module.
        x (Union[np.ndarray, float, int]): Input value or array.
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated modified Bessel I0 array.
    """
    return backend_module.array(sc.i0(x))


@numpy_eager_registry.register("BesselI1")
def _np_bessel_i1(
    backend_module: ModuleType,
    x: Union[np.ndarray, float, int],
    **kwargs: Union[float, int, str, bool, None],
) -> np.ndarray:
    """Evaluate modified Bessel function of the first kind of order 1.

    Args:
        backend_module (ModuleType): Active backend module.
        x (Union[np.ndarray, float, int]): Input value or array.
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated modified Bessel I1 array.
    """
    return backend_module.array(sc.i1(x))


@numpy_eager_registry.register("BesselJn")
def _np_bessel_jn(
    backend_module: ModuleType,
    x: Union[np.ndarray, float, int],
    y: Union[np.ndarray, float, int],
    **kwargs: Union[float, int, str, bool, None],
) -> np.ndarray:
    """Evaluate Bessel function of the first kind of real order.

    Args:
        backend_module (ModuleType): Active backend module.
        x (Union[np.ndarray, float, int]): Order of Bessel function.
        y (Union[np.ndarray, float, int]): Argument of Bessel function.
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated Bessel Jn array.
    """
    return backend_module.array(sc.jv(x, y))


@numpy_eager_registry.register("Bartlett")
def _np_bartlett(
    backend_module: ModuleType,
    M: int,
    **kwargs: Union[float, int, str, bool, None],
) -> np.ndarray:
    """Return the Bartlett window.

    Args:
        backend_module (ModuleType): Active backend module.
        M (int): Number of points in the output window.
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: The triangular Bartlett window.
    """
    return np.bartlett(M)
