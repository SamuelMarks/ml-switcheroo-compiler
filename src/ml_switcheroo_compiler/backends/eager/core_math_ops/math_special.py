# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Core utilities."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry


@global_eager_registry.register("Erf")
def _erf(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _erf operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    x: object = args[0]
    if hasattr(backend_module, "erf"):
        return backend_module.erf(x)
    # A&S approximation 7.1.26
    # erf(x) = 1 - (a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5)*e^{-x^2}
    p: object = 0.3275911
    a1: object = 0.254829592
    a2: object = -0.284496736
    a3: object = 1.421413741
    a4: object = -1.453152027
    a5: object = 1.061405429

    sign: object = backend_module.sign(x)
    abs_x: object = backend_module.abs(x)
    t: object = 1.0 / (1.0 + p * abs_x)
    y: object = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * backend_module.exp(-abs_x * abs_x)
    return sign * y


@global_eager_registry.register("Erfc")
def _erfc(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _erfc operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    x: object = args[0]
    if hasattr(backend_module, "erfc"):
        return backend_module.erfc(x)
    return 1.0 - _erf(backend_module, x)


@global_eager_registry.register("Erfinv")
def _erfinv(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _erfinv operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    x: object = args[0]
    if hasattr(backend_module, "erfinv"):
        return backend_module.erfinv(x)
    # Approximation of erfinv
    # https://en.wikipedia.org/wiki/Error_function#Approximation_with_elementary_functions
    # For a naive approximation:
    a: object = 0.147
    ln_1_x2: object = backend_module.log(1.0 - x * x + 1e-12)
    term1: object = 2.0 / (3.141592653589793 * a) + ln_1_x2 / 2.0
    term2: object = ln_1_x2 / a
    return backend_module.sign(x) * backend_module.sqrt(backend_module.sqrt(term1 * term1 - term2) - term1)


@global_eager_registry.register("BesselI0")
def _bessel_i0(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _bessel_i0 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import scipy.special

    return scipy.special.i0(*args, **kwargs)


@global_eager_registry.register("BesselI0e")
def _bessel_i0e(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _bessel_i0e operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import scipy.special

    return scipy.special.i0e(*args, **kwargs)


@global_eager_registry.register("BesselI1")
def _bessel_i1(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _bessel_i1 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import scipy.special

    return scipy.special.i1(*args, **kwargs)


@global_eager_registry.register("BesselI1e")
def _bessel_i1e(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _bessel_i1e operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import scipy.special

    return scipy.special.i1e(*args, **kwargs)


@global_eager_registry.register("BesselJ0")
def _bessel_j0(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _bessel_j0 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import scipy.special

    return scipy.special.j0(*args, **kwargs)


@global_eager_registry.register("BesselJ1")
def _bessel_j1(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _bessel_j1 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import scipy.special

    return scipy.special.j1(*args, **kwargs)


@global_eager_registry.register("BesselJn")
def _bessel_jn(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _bessel_jn operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import scipy.special

    return scipy.special.jv(*args, **kwargs)


@global_eager_registry.register("BesselK0")
def _bessel_k0(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _bessel_k0 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import scipy.special

    return scipy.special.k0(*args, **kwargs)


@global_eager_registry.register("BesselK0e")
def _bessel_k0e(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _bessel_k0e operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import scipy.special

    return scipy.special.k0e(*args, **kwargs)


@global_eager_registry.register("BesselK1")
def _bessel_k1(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _bessel_k1 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import scipy.special

    return scipy.special.k1(*args, **kwargs)


@global_eager_registry.register("BesselK1e")
def _bessel_k1e(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _bessel_k1e operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import scipy.special

    return scipy.special.k1e(*args, **kwargs)


@global_eager_registry.register("BesselY0")
def _bessel_y0(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _bessel_y0 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import scipy.special

    return scipy.special.y0(*args, **kwargs)


@global_eager_registry.register("BesselY1")
def _bessel_y1(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _bessel_y1 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import scipy.special

    return scipy.special.y1(*args, **kwargs)


@global_eager_registry.register("Digamma")
def _digamma(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _digamma operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import scipy.special

    return scipy.special.digamma(*args, **kwargs)


@global_eager_registry.register("Igammac")
def _igammac(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _igammac operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import scipy.special

    return scipy.special.gammaincc(*args, **kwargs)


@global_eager_registry.register("Polygamma")
def _polygamma(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _polygamma operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import scipy.special

    return scipy.special.polygamma(*args, **kwargs)


@global_eager_registry.register("Igamma")
def _igamma(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _igamma operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import scipy.special

    return scipy.special.gammainc(*args, **kwargs)


@global_eager_registry.register("Gamma")
def _gamma(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _gamma operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import scipy.special

    return scipy.special.gamma(*args, **kwargs)


@global_eager_registry.register("ModifiedBesselI1")
def _np_modifiedbesseli1(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_modifiedbesseli1 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import scipy.special

    return scipy.special.i1(*args, **kwargs)
