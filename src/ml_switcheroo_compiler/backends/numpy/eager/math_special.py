# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Numpy eager special math functions."""

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("BesselJ0")
def _np_bessel_j0(backend_module, *args, **kwargs):
    """Evaluate _np_bessel_j0 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import scipy.special as sc

    x = args[0] if args else kwargs.get("x", 0.0)
    return backend_module.array(sc.j0(x))


@numpy_eager_registry.register("BesselJ1")
def _np_bessel_j1(backend_module, *args, **kwargs):
    """Evaluate _np_bessel_j1 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import scipy.special as sc

    x = args[0] if args else kwargs.get("x", 0.0)
    return backend_module.array(sc.j1(x))


@numpy_eager_registry.register("BesselK0")
def _np_bessel_k0(backend_module, *args, **kwargs):
    """Evaluate _np_bessel_k0 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import scipy.special as sc

    x = args[0] if args else kwargs.get("x", 0.0)
    return backend_module.array(sc.k0(x))


@numpy_eager_registry.register("BesselK0e")
def _np_bessel_k0e(backend_module, *args, **kwargs):
    """Evaluate _np_bessel_k0e operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import scipy.special as sc

    x = args[0] if args else kwargs.get("x", 0.0)
    return backend_module.array(sc.k0e(x))


@numpy_eager_registry.register("BesselK1")
def _np_bessel_k1(backend_module, *args, **kwargs):
    """Evaluate _np_bessel_k1 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import scipy.special as sc

    x = args[0] if args else kwargs.get("x", 0.0)
    return backend_module.array(sc.k1(x))


@numpy_eager_registry.register("BesselK1e")
def _np_bessel_k1e(backend_module, *args, **kwargs):
    """Evaluate _np_bessel_k1e operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import scipy.special as sc

    x = args[0] if args else kwargs.get("x", 0.0)
    return backend_module.array(sc.k1e(x))


@numpy_eager_registry.register("BesselY0")
def _np_bessel_y0(backend_module, *args, **kwargs):
    """Evaluate _np_bessel_y0 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import scipy.special as sc

    x = args[0] if args else kwargs.get("x", 0.0)
    return backend_module.array(sc.y0(x))


@numpy_eager_registry.register("BesselY1")
def _np_bessel_y1(backend_module, *args, **kwargs):
    """Evaluate _np_bessel_y1 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import scipy.special as sc

    x = args[0] if args else kwargs.get("x", 0.0)
    return backend_module.array(sc.y1(x))


@numpy_eager_registry.register("Dawsn")
def _np_dawsn(backend_module, *args, **kwargs):
    """Evaluate _np_dawsn operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import scipy.special as sc

    x = args[0] if args else kwargs.get("x", 0.0)
    return backend_module.array(sc.dawsn(x))


@numpy_eager_registry.register("Expint")
def _np_expint(backend_module, *args, **kwargs):
    """Evaluate _np_expint operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import scipy.special as sc

    x = args[0] if args else kwargs.get("x", 0.0)
    n = kwargs.get("n", 1)
    if len(args) > 1:
        n = args[1]
    return backend_module.array(sc.expn(n, x))


@numpy_eager_registry.register("FresnelCos")
def _np_fresnel_cos(backend_module, *args, **kwargs):
    """Evaluate _np_fresnel_cos operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import scipy.special as sc

    x = args[0] if args else kwargs.get("x", 0.0)
    return backend_module.array(sc.fresnel(x)[1])


@numpy_eager_registry.register("FresnelSin")
def _np_fresnel_sin(backend_module, *args, **kwargs):
    """Evaluate _np_fresnel_sin operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import scipy.special as sc

    x = args[0] if args else kwargs.get("x", 0.0)
    return backend_module.array(sc.fresnel(x)[0])


@numpy_eager_registry.register("Spence")
def _np_spence(backend_module, *args, **kwargs):
    """Evaluate _np_spence operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import scipy.special as sc

    x = args[0] if args else kwargs.get("x", 0.0)
    return backend_module.array(sc.spence(x))


@numpy_eager_registry.register("BesselI0")
def _np_bessel_i0(backend_module, x, **kwargs):
    """Evaluate _np_bessel_i0 operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import scipy.special as sc

    return backend_module.array(sc.i0(x))


@numpy_eager_registry.register("BesselI1")
def _np_bessel_i1(backend_module, x, **kwargs):
    """Evaluate _np_bessel_i1 operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import scipy.special as sc

    return backend_module.array(sc.i1(x))


@numpy_eager_registry.register("BesselJn")
def _np_bessel_jn(backend_module, x, y, **kwargs):
    """Evaluate _np_bessel_jn operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        y (object): The y parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import scipy.special as sc

    return backend_module.array(sc.jv(x, y))


@numpy_eager_registry.register("Bartlett")
def _np_bartlett(backend_module, M, **kwargs):
    """Evaluate _np_bartlett operation.

    Args:
        backend_module (object): The backend_module parameter.
        M (object): The M parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return np.bartlett(M)
