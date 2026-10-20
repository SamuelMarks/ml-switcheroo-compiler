"""Numpy eager special math functions."""

from __future__ import annotations

import importlib
import typing
from types import ModuleType

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, numpy_eager_registry
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol


def _sc() -> ModuleType:
    """Import and return `scipy.special` dynamically.

    Returns:
        ModuleType: The `scipy.special` module.
    """
    return importlib.import_module("scipy.special")


@numpy_eager_registry.register("BesselJ0")
def _np_bessel_j0(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Evaluate Bessel function of the first kind of order 0.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int]): Positional arguments (x).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated Bessel function array.
    """
    x = args[0] if args else kwargs.get("x", 0.0)
    return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], backend_module.array(_sc().j0(x)))


@numpy_eager_registry.register("BesselJ1")
def _np_bessel_j1(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Evaluate Bessel function of the first kind of order 1.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int]): Positional arguments (x).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated Bessel function array.
    """
    x = args[0] if args else kwargs.get("x", 0.0)
    return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], backend_module.array(_sc().j1(x)))


@numpy_eager_registry.register("BesselK0")
def _np_bessel_k0(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Evaluate modified Bessel function of the second kind of order 0.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int]): Positional arguments (x).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated modified Bessel function array.
    """
    x = args[0] if args else kwargs.get("x", 0.0)
    return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], backend_module.array(_sc().k0(x)))


@numpy_eager_registry.register("BesselK0e")
def _np_bessel_k0e(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Evaluate exponentially scaled modified Bessel function K0e.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int]): Positional arguments (x).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated exponentially scaled Bessel array.
    """
    x = args[0] if args else kwargs.get("x", 0.0)
    return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], backend_module.array(_sc().k0e(x)))


@numpy_eager_registry.register("BesselK1")
def _np_bessel_k1(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Evaluate modified Bessel function of the second kind of order 1.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int]): Positional arguments (x).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated modified Bessel function array.
    """
    x = args[0] if args else kwargs.get("x", 0.0)
    return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], backend_module.array(_sc().k1(x)))


@numpy_eager_registry.register("BesselK1e")
def _np_bessel_k1e(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Evaluate exponentially scaled modified Bessel function K1e.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int]): Positional arguments (x).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated exponentially scaled Bessel array.
    """
    x = args[0] if args else kwargs.get("x", 0.0)
    return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], backend_module.array(_sc().k1e(x)))


@numpy_eager_registry.register("BesselY0")
def _np_bessel_y0(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Evaluate Bessel function of the second kind of order 0.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int]): Positional arguments (x).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated Bessel Y0 array.
    """
    x = args[0] if args else kwargs.get("x", 0.0)
    return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], backend_module.array(_sc().y0(x)))


@numpy_eager_registry.register("BesselY1")
def _np_bessel_y1(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Evaluate Bessel function of the second kind of order 1.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int]): Positional arguments (x).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated Bessel Y1 array.
    """
    x = args[0] if args else kwargs.get("x", 0.0)
    return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], backend_module.array(_sc().y1(x)))


@numpy_eager_registry.register("Dawsn")
def _np_dawsn(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Evaluate Dawson's integral.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int]): Positional arguments (x).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated Dawson's integral array.
    """
    x = args[0] if args else kwargs.get("x", 0.0)
    return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], backend_module.array(_sc().dawsn(x)))


@numpy_eager_registry.register("Expint")
def _np_expint(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Evaluate exponential integral E_n(x).

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int]): Positional arguments (x, n).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated exponential integral array.
    """
    x = args[0] if args else kwargs.get("x", 0.0)
    n: EagerValue = kwargs.get("n", 1)
    if len(args) > 1:
        n = args[1]
    return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], backend_module.array(_sc().expn(n, x)))


@numpy_eager_registry.register("FresnelCos")
def _np_fresnel_cos(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Evaluate Fresnel cosine integral.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int]): Positional arguments (x).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated Fresnel cosine integral array.
    """
    x = args[0] if args else kwargs.get("x", 0.0)
    return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], backend_module.array(_sc().fresnel(x)[1]))


@numpy_eager_registry.register("FresnelSin")
def _np_fresnel_sin(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Evaluate Fresnel sine integral.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int]): Positional arguments (x).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated Fresnel sine integral array.
    """
    x = args[0] if args else kwargs.get("x", 0.0)
    return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], backend_module.array(_sc().fresnel(x)[0]))


@numpy_eager_registry.register("Spence")
def _np_spence(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Evaluate Spence's function (dilogarithm).

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int]): Positional arguments (x).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated Spence's function array.
    """
    x = args[0] if args else kwargs.get("x", 0.0)
    return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], backend_module.array(_sc().spence(x)))


@numpy_eager_registry.register("BesselI0")
def _np_bessel_i0(
    backend_module: BackendModuleProtocol,
    x: np.ndarray[tuple[int, ...], np.dtype[np.generic]] | float | int,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Evaluate modified Bessel function of the first kind of order 0.

    Args:
        backend_module (ModuleType): Active backend module.
        x (Union[np.ndarray, float, int]): Input value or array.
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated modified Bessel I0 array.
    """
    return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], backend_module.array(_sc().i0(x)))


@numpy_eager_registry.register("BesselI1")
def _np_bessel_i1(
    backend_module: BackendModuleProtocol,
    x: np.ndarray[tuple[int, ...], np.dtype[np.generic]] | float | int,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Evaluate modified Bessel function of the first kind of order 1.

    Args:
        backend_module (ModuleType): Active backend module.
        x (Union[np.ndarray, float, int]): Input value or array.
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated modified Bessel I1 array.
    """
    return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], backend_module.array(_sc().i1(x)))


@numpy_eager_registry.register("BesselJn")
def _np_bessel_jn(
    backend_module: BackendModuleProtocol,
    x: np.ndarray[tuple[int, ...], np.dtype[np.generic]] | float | int,
    y: np.ndarray[tuple[int, ...], np.dtype[np.generic]] | float | int,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Evaluate Bessel function of the first kind of real order.

    Args:
        backend_module (ModuleType): Active backend module.
        x (Union[np.ndarray, float, int]): Order of Bessel function.
        y (Union[np.ndarray, float, int]): Argument of Bessel function.
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated Bessel Jn array.
    """
    return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], backend_module.array(_sc().jv(x, y)))


@numpy_eager_registry.register("Bartlett")
def _np_bartlett(
    backend_module: BackendModuleProtocol,
    M: int,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Return the Bartlett window.

    Args:
        backend_module (ModuleType): Active backend module.
        M (int): Number of points in the output window.
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: The triangular Bartlett window.
    """
    return np.bartlett(M)


@numpy_eager_registry.register("Digamma")
def _np_digamma(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Evaluate the digamma (psi) function via `scipy.special.digamma`.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int]): Positional arguments (x).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated digamma function array.
    """
    return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], backend_module.asarray(_sc().digamma(*args, **kwargs)))


@numpy_eager_registry.register("Igammac")
def _np_igammac(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Evaluate regularized upper incomplete gamma function via `scipy.special.gammaincc`.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int]): Positional arguments (a, x).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated regularized upper incomplete gamma array.
    """
    return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], backend_module.asarray(_sc().gammaincc(*args, **kwargs)))


@numpy_eager_registry.register("Polygamma")
def _np_polygamma(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Evaluate polygamma function of order n via `scipy.special.polygamma`.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int]): Positional arguments (n, x).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated polygamma array.
    """
    return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], backend_module.asarray(_sc().polygamma(*args, **kwargs)))


@numpy_eager_registry.register("Igamma")
def _np_igamma(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Evaluate regularized lower incomplete gamma function via `scipy.special.gammainc`.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int]): Positional arguments (a, x).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated regularized lower incomplete gamma array.
    """
    return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], backend_module.asarray(_sc().gammainc(*args, **kwargs)))


@numpy_eager_registry.register("Gamma")
def _np_gamma(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Evaluate the gamma function via `scipy.special.gamma`.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int]): Positional arguments (x).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated gamma function array.
    """
    return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], backend_module.asarray(_sc().gamma(*args, **kwargs)))


@numpy_eager_registry.register("ModifiedBesselI1")
def _np_modifiedbesseli1(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Evaluate modified Bessel function of order 1 via `scipy.special.i1`.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int]): Positional arguments (x).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated modified Bessel I1 array.
    """
    if (args and args[0] is None) or kwargs.get("x", 0) is None:
        raise TypeError("ModifiedBesselI1 input cannot be None")
    return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], backend_module.asarray(_sc().i1(*args, **kwargs)))


@numpy_eager_registry.register("Betainc")
def _np_betainc_special(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Evaluate regularized incomplete beta function via `scipy.special.betainc`.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int]): Positional arguments (a, b, x).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated betainc array.
    """
    return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], backend_module.asarray(_sc().betainc(*args, **kwargs)))


@numpy_eager_registry.register("Beta")
def _np_beta_func(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Evaluate the Euler Beta function via `scipy.special.beta`.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int]): Positional arguments (a, b).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated Beta function array.
    """
    if len(args) >= 3 and args[0] is None:
        args = args[1:]
    return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], backend_module.asarray(_sc().beta(*args, **kwargs)))


@numpy_eager_registry.register("BetaPdf")
def _np_betapdf(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Evaluate the Beta probability density function via `scipy.stats.beta.pdf`.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int]): Positional arguments (x, a, b).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated Beta PDF array.
    """
    import scipy.stats as stats

    return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], backend_module.asarray(stats.beta.pdf(*args, **kwargs)))


@numpy_eager_registry.register("Zeta")
def _np_zeta(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Evaluate the Hurwitz/Riemann zeta function via `scipy.special.zeta`.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int]): Positional arguments (x, q).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated zeta function array.
    """
    return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], backend_module.asarray(_sc().zeta(*args, **kwargs)))


@numpy_eager_registry.register("Xlog1Py")
def _np_xlog1py(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Evaluate `x * log1p(y)` with safe zero handling via `scipy.special.xlog1py`.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, float, int]): Positional arguments (x, y).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated `x * log1p(y)` array.
    """
    return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], backend_module.asarray(_sc().xlog1py(*args, **kwargs)))
