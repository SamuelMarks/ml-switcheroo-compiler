"""Thin NumPy/SciPy special-function eager kernel bindings."""

from __future__ import annotations

import importlib
import types
from types import ModuleType

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    numpy_eager_registry,
)
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module


@numpy_eager_registry.register("Betainc")
def _np_betainc(backend_module: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Execute the eager np betainc operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.

    Raises:
        ImportError: When raised during `_np_betainc` (scipy is required for Betainc in numpy eager backend).
    """
    try:
        import scipy.special

        return adapt_backend_module(backend_module).require_symbol("array")(scipy.special.betainc(*args, **kwargs))
    except ImportError as e:
        raise ImportError("scipy is required for Betainc in numpy eager backend") from e


@numpy_eager_registry.register("EinsumPath")
def _np_einsum_path_(backend_module: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Implement EinsumPath via einsum_path.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        np.ndarray: The computed result.

    Raises:
        TypeError: If the first subscript argument is a boolean array.
    """
    if args and isinstance(args[0], np.ndarray) and args[0].dtype == np.bool_:
        raise TypeError("einsum_path subscripts cannot be a boolean array")
    return adapt_backend_module(backend_module).require_symbol("einsum_path")(*args, **kwargs)


def _get_sc() -> ModuleType | None:
    """Import `scipy.special` if available, returning `None` otherwise.

    Returns:
        ModuleType | None: Imported `scipy.special` module or `None`.
    """
    try:
        return importlib.import_module("scipy.special")
    except ImportError:
        return None


def _sc() -> ModuleType:
    """Import and return `scipy.special` dynamically.

    Returns:
        ModuleType: The `scipy.special` module.
    """
    return importlib.import_module("scipy.special")


@numpy_eager_registry.register("BesselJ0")
def _np_bessel_j0(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Evaluate Bessel function of the first kind of order 0.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments (x).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated Bessel function array.
    """
    x = args[0] if args else kwargs.get("x", 0.0)
    return ensure_ndarray(adapt_backend_module(backend_module).require_symbol("array")(_sc().j0(x)))


@numpy_eager_registry.register("BesselJ1")
def _np_bessel_j1(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Evaluate Bessel function of the first kind of order 1.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments (x).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated Bessel function array.
    """
    x = args[0] if args else kwargs.get("x", 0.0)
    return ensure_ndarray(adapt_backend_module(backend_module).require_symbol("array")(_sc().j1(x)))


@numpy_eager_registry.register("BesselK0")
def _np_bessel_k0(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Evaluate modified Bessel function of the second kind of order 0.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments (x).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated modified Bessel function array.
    """
    x = args[0] if args else kwargs.get("x", 0.0)
    return ensure_ndarray(adapt_backend_module(backend_module).require_symbol("array")(_sc().k0(x)))


@numpy_eager_registry.register("BesselK0e")
def _np_bessel_k0e(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Evaluate exponentially scaled modified Bessel function K0e.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments (x).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated exponentially scaled Bessel array.
    """
    x = args[0] if args else kwargs.get("x", 0.0)
    return ensure_ndarray(adapt_backend_module(backend_module).require_symbol("array")(_sc().k0e(x)))


@numpy_eager_registry.register("BesselK1")
def _np_bessel_k1(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Evaluate modified Bessel function of the second kind of order 1.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments (x).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated modified Bessel function array.
    """
    x = args[0] if args else kwargs.get("x", 0.0)
    return ensure_ndarray(adapt_backend_module(backend_module).require_symbol("array")(_sc().k1(x)))


@numpy_eager_registry.register("BesselK1e")
def _np_bessel_k1e(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Evaluate exponentially scaled modified Bessel function K1e.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments (x).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated exponentially scaled Bessel array.
    """
    x = args[0] if args else kwargs.get("x", 0.0)
    return ensure_ndarray(adapt_backend_module(backend_module).require_symbol("array")(_sc().k1e(x)))


@numpy_eager_registry.register("BesselY0")
def _np_bessel_y0(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Evaluate Bessel function of the second kind of order 0.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments (x).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated Bessel Y0 array.
    """
    x = args[0] if args else kwargs.get("x", 0.0)
    return ensure_ndarray(adapt_backend_module(backend_module).require_symbol("array")(_sc().y0(x)))


@numpy_eager_registry.register("BesselY1")
def _np_bessel_y1(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Evaluate Bessel function of the second kind of order 1.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments (x).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated Bessel Y1 array.
    """
    x = args[0] if args else kwargs.get("x", 0.0)
    return ensure_ndarray(adapt_backend_module(backend_module).require_symbol("array")(_sc().y1(x)))


@numpy_eager_registry.register("Dawsn")
def _np_dawsn(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Evaluate Dawson's integral.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments (x).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated Dawson's integral array.
    """
    x = args[0] if args else kwargs.get("x", 0.0)
    return ensure_ndarray(adapt_backend_module(backend_module).require_symbol("array")(_sc().dawsn(x)))


@numpy_eager_registry.register("Expint")
def _np_expint(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Evaluate exponential integral E_n(x).

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments (x, n).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated exponential integral array.
    """
    x = args[0] if args else kwargs.get("x", 0.0)
    n: BackendArg = kwargs.get("n", 1)
    if len(args) > 1:
        n = args[1]
    return ensure_ndarray(adapt_backend_module(backend_module).require_symbol("array")(_sc().expn(n, x)))


@numpy_eager_registry.register("FresnelCos")
def _np_fresnel_cos(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Evaluate Fresnel cosine integral.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments (x).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated Fresnel cosine integral array.
    """
    x = args[0] if args else kwargs.get("x", 0.0)
    return ensure_ndarray(adapt_backend_module(backend_module).require_symbol("array")(_sc().fresnel(x)[1]))


@numpy_eager_registry.register("FresnelSin")
def _np_fresnel_sin(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Evaluate Fresnel sine integral.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments (x).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated Fresnel sine integral array.
    """
    x = args[0] if args else kwargs.get("x", 0.0)
    return ensure_ndarray(adapt_backend_module(backend_module).require_symbol("array")(_sc().fresnel(x)[0]))


@numpy_eager_registry.register("Spence")
def _np_spence(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Evaluate Spence's function (dilogarithm).

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments (x).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated Spence's function array.
    """
    x = args[0] if args else kwargs.get("x", 0.0)
    return ensure_ndarray(adapt_backend_module(backend_module).require_symbol("array")(_sc().spence(x)))


@numpy_eager_registry.register("BesselI0")
def _np_bessel_i0(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    x: np.ndarray[tuple[int, ...], np.dtype[np.float64]] | float | int,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Evaluate modified Bessel function of the first kind of order 0.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        x (np.ndarray[tuple[int, ...], np.dtype[np.float64]] | float | int): Input value or array.
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated modified Bessel I0 array.
    """
    return ensure_ndarray(adapt_backend_module(backend_module).require_symbol("array")(_sc().i0(x)))


@numpy_eager_registry.register("BesselI1")
def _np_bessel_i1(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    x: np.ndarray[tuple[int, ...], np.dtype[np.float64]] | float | int,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Evaluate modified Bessel function of the first kind of order 1.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        x (np.ndarray[tuple[int, ...], np.dtype[np.float64]] | float | int): Input value or array.
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated modified Bessel I1 array.
    """
    return ensure_ndarray(adapt_backend_module(backend_module).require_symbol("array")(_sc().i1(x)))


@numpy_eager_registry.register("BesselJn")
def _np_bessel_jn(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    x: np.ndarray[tuple[int, ...], np.dtype[np.float64]] | float | int,
    y: np.ndarray[tuple[int, ...], np.dtype[np.float64]] | float | int,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Evaluate Bessel function of the first kind of real order.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        x (np.ndarray[tuple[int, ...], np.dtype[np.float64]] | float | int): Order of Bessel function.
        y (np.ndarray[tuple[int, ...], np.dtype[np.float64]] | float | int): Argument of Bessel function.
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated Bessel Jn array.
    """
    return ensure_ndarray(adapt_backend_module(backend_module).require_symbol("array")(_sc().jv(x, y)))


@numpy_eager_registry.register("Digamma")
def _np_digamma(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Evaluate the digamma (psi) function via `scipy.special.digamma`.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments (x).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated digamma function array.
    """
    return ensure_ndarray(
        adapt_backend_module(backend_module).require_symbol("asarray")(_sc().digamma(*args, **kwargs))
    )


@numpy_eager_registry.register("Polygamma")
def _np_polygamma(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Evaluate polygamma function of order n via `scipy.special.polygamma`.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments (n, x).
        **kwargs (BackendArg): Keyword arguments.

    Returns:
        np.ndarray: Evaluated polygamma array.
    """
    n = args[0]
    if len(args) > 1:
        x = args[1]
    elif "x" in kwargs:
        x = kwargs["x"]
    else:
        return ensure_ndarray(adapt_backend_module(backend_module).require_symbol("zeros_like")(n))
    return ensure_ndarray(adapt_backend_module(backend_module).require_symbol("array")(_sc().polygamma(n, x)))


@numpy_eager_registry.register("Gamma")
def _np_gamma(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Evaluate the gamma function via `scipy.special.gamma`.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments (x).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated gamma function array.
    """
    return ensure_ndarray(adapt_backend_module(backend_module).require_symbol("asarray")(_sc().gamma(*args, **kwargs)))


@numpy_eager_registry.register("ModifiedBesselI1")
def _np_modifiedbesseli1(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Evaluate modified Bessel function of order 1 via `scipy.special.i1`.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments (x).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated modified Bessel I1 array.

    Raises:
        TypeError: When raised during `_np_modifiedbesseli1` (ModifiedBesselI1 input cannot be None).
    """
    if (args and args[0] is None) or kwargs.get("x", 0) is None:
        raise TypeError("ModifiedBesselI1 input cannot be None")
    return ensure_ndarray(adapt_backend_module(backend_module).require_symbol("asarray")(_sc().i1(*args, **kwargs)))


@numpy_eager_registry.register("Betainc")
def _np_betainc_special(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Evaluate regularized incomplete beta function via `scipy.special.betainc`.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments (a, b, x).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated betainc array.
    """
    return ensure_ndarray(
        adapt_backend_module(backend_module).require_symbol("asarray")(_sc().betainc(*args, **kwargs))
    )


@numpy_eager_registry.register("Beta")
def _np_beta_func(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Evaluate the Euler Beta function via `scipy.special.beta`.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments (a, b).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated Beta function array.
    """
    if len(args) >= 3 and args[0] is None:
        args = args[1:]
    return ensure_ndarray(adapt_backend_module(backend_module).require_symbol("asarray")(_sc().beta(*args, **kwargs)))


@numpy_eager_registry.register("BetaPdf")
def _np_betapdf(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Evaluate the Beta probability density function via `scipy.stats.beta.pdf`.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments (x, a, b).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated Beta PDF array.
    """
    import scipy.stats as stats

    return ensure_ndarray(
        adapt_backend_module(backend_module).require_symbol("asarray")(stats.beta.pdf(*args, **kwargs))
    )


@numpy_eager_registry.register("Zeta")
def _np_zeta(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Evaluate the Hurwitz/Riemann zeta function via `scipy.special.zeta`.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments (x, optional q).
        **kwargs (BackendArg): Keyword arguments (optional `q`).

    Returns:
        np.ndarray: Evaluated zeta array.
    """
    sc = _sc()
    x = args[0]
    if len(args) > 1:
        q = args[1]
    elif "q" in kwargs:
        q = kwargs["q"]
    else:
        return ensure_ndarray(adapt_backend_module(backend_module).require_symbol("zeros_like")(x))
    return ensure_ndarray(adapt_backend_module(backend_module).require_symbol("array")(sc.zeta(x, q)))


@numpy_eager_registry.register("Xlog1Py")
def _np_xlog1py(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Evaluate `x * log1p(y)` with safe zero handling via `scipy.special.xlog1py`.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments (x, y).
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Evaluated `x * log1p(y)` array.
    """
    return ensure_ndarray(
        adapt_backend_module(backend_module).require_symbol("asarray")(_sc().xlog1py(*args, **kwargs))
    )


@numpy_eager_registry.register("Erfinv")
def _np_erfinv(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np erfinv operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    import scipy.special as sc

    return adapt_backend_module(backend_module).require_symbol("array")(sc.erfinv(args[0]))


@numpy_eager_registry.register("Igamma")
def _np_igamma(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np igamma operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    import scipy.special as sc

    a = args[0]
    x = args[1] if len(args) > 1 else kwargs.get("x")
    return adapt_backend_module(backend_module).require_symbol("array")(sc.gammainc(a, x))


@numpy_eager_registry.register("Igammac")
def _np_igammac(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np igammac operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    import scipy.special as sc

    a = args[0]
    x = args[1] if len(args) > 1 else kwargs.get("x")
    return adapt_backend_module(backend_module).require_symbol("array")(sc.gammaincc(a, x))
