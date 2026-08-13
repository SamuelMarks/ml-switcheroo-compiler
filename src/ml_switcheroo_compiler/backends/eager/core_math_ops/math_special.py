# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Core utilities."""

from __future__ import annotations

from typing import Any

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry


@global_eager_registry.register("Erf")
def _erf(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _erf operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    x = args[0]
    if hasattr(backend_module, "erf"):
        return backend_module.erf(x)
    # A&S approximation 7.1.26
    # erf(x) = 1 - (a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5)*e^{-x^2}
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429

    sign = backend_module.sign(x)
    abs_x = backend_module.abs(x)
    t = 1.0 / (1.0 + p * abs_x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * backend_module.exp(-abs_x * abs_x)
    return sign * y


@global_eager_registry.register("Erfc")
def _erfc(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _erfc operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    x = args[0]
    if hasattr(backend_module, "erfc"):
        return backend_module.erfc(x)
    return 1.0 - _erf(backend_module, x)


@global_eager_registry.register("Erfinv")
def _erfinv(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _erfinv operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    x = args[0]
    if hasattr(backend_module, "erfinv"):
        return backend_module.erfinv(x)
    # Approximation of erfinv
    # https://en.wikipedia.org/wiki/Error_function#Approximation_with_elementary_functions
    # For a naive approximation:
    a = 0.147
    ln_1_x2 = backend_module.log(1.0 - x * x + 1e-12)
    term1 = 2.0 / (3.141592653589793 * a) + ln_1_x2 / 2.0
    term2 = ln_1_x2 / a
    return backend_module.sign(x) * backend_module.sqrt(backend_module.sqrt(term1 * term1 - term2) - term1)


@global_eager_registry.register("BesselI0")
def _bessel_i0(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _bessel_i0 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "i0", None)
    if func:
        return func(*args, **kwargs)
    return None


@global_eager_registry.register("BesselI0e")
def _bessel_i0e(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _bessel_i0e operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "i0e", None)
    if func:
        return func(*args, **kwargs)
    return None


@global_eager_registry.register("BesselI1")
def _bessel_i1(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _bessel_i1 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "i1", None)
    if func:
        return func(*args, **kwargs)
    return None


@global_eager_registry.register("BesselI1e")
def _bessel_i1e(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _bessel_i1e operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "i1e", None)
    if func:
        return func(*args, **kwargs)
    return None


@global_eager_registry.register("BesselJ0")
def _bessel_j0(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _bessel_j0 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "j0", None)
    if func:
        return func(*args, **kwargs)
    return None


@global_eager_registry.register("BesselJ1")
def _bessel_j1(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _bessel_j1 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "j1", None)
    if func:
        return func(*args, **kwargs)
    return None


@global_eager_registry.register("BesselJn")
def _bessel_jn(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _bessel_jn operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "jv", getattr(backend_module, "jn", None))
    if func:
        return func(*args, **kwargs)
    return None


@global_eager_registry.register("BesselK0")
def _bessel_k0(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _bessel_k0 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "k0", None)
    if func:
        return func(*args, **kwargs)
    return None


@global_eager_registry.register("BesselK0e")
def _bessel_k0e(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _bessel_k0e operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "k0e", None)
    if func:
        return func(*args, **kwargs)
    return None


@global_eager_registry.register("BesselK1")
def _bessel_k1(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _bessel_k1 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "k1", None)
    if func:
        return func(*args, **kwargs)
    return None


@global_eager_registry.register("BesselK1e")
def _bessel_k1e(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _bessel_k1e operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "k1e", None)
    if func:
        return func(*args, **kwargs)
    return None


@global_eager_registry.register("BesselY0")
def _bessel_y0(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _bessel_y0 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "y0", None)
    if func:
        return func(*args, **kwargs)
    return None


@global_eager_registry.register("BesselY1")
def _bessel_y1(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _bessel_y1 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "y1", None)
    if func:
        return func(*args, **kwargs)
    return None


@global_eager_registry.register("Digamma")
def _digamma(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _digamma operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "digamma", getattr(backend_module, "psi", None))
    if func:
        return func(*args, **kwargs)
    return None


@global_eager_registry.register("Igammac")
def _igammac(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _igammac operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "igammac", getattr(backend_module, "gammaincc", None))
    if func:
        return func(*args, **kwargs)
    return None


@global_eager_registry.register("Polygamma")
def _polygamma(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _polygamma operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "polygamma", None)
    if func:
        return func(*args, **kwargs)
    return None


@global_eager_registry.register("Igamma")
def _igamma(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _igamma operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "igamma", getattr(backend_module, "gammainc", None))
    if func:
        return func(*args, **kwargs)
    return None


@global_eager_registry.register("Gamma")
def _gamma(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _gamma operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    import math

    func = getattr(backend_module, "gamma", math.gamma)
    return func(*args, **kwargs)


@global_eager_registry.register("Gamma")
def _np_gamma(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_gamma operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "gamma", getattr(backend_module, "gamma", None))
    if func is not None:
        try:
            return func(*args, **kwargs)
        except Exception:
            pass
    import math

    import numpy as np

    return np.array([math.gamma(x) for x in args[0].flatten()]).reshape(args[0].shape)


@global_eager_registry.register("ModifiedBesselI1")
def _np_modifiedbesseli1(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_modifiedbesseli1 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "modifiedbesseli1", getattr(backend_module, "modifiedbesseli1", None))
    if func is not None:
        try:
            return func(*args, **kwargs)
        except Exception:
            pass
    import numpy as np

    return args[0]
