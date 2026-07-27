# ruff: noqa: E501
"""Numpy eager special math functions."""

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("BesselJ0")
def _np_bessel_j0(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the bessel j0 logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    import scipy.special as sc

    x = args[0] if args else kwargs.get("x", 0.0)
    return backend_module.array(sc.j0(x))


@numpy_eager_registry.register("BesselJ1")
def _np_bessel_j1(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the bessel j1 logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    import scipy.special as sc

    x = args[0] if args else kwargs.get("x", 0.0)
    return backend_module.array(sc.j1(x))


@numpy_eager_registry.register("BesselK0")
def _np_bessel_k0(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the bessel k0 logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    import scipy.special as sc

    x = args[0] if args else kwargs.get("x", 0.0)
    return backend_module.array(sc.k0(x))


@numpy_eager_registry.register("BesselK0e")
def _np_bessel_k0e(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the bessel k0e logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    import scipy.special as sc

    x = args[0] if args else kwargs.get("x", 0.0)
    return backend_module.array(sc.k0e(x))


@numpy_eager_registry.register("BesselK1")
def _np_bessel_k1(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the bessel k1 logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    import scipy.special as sc

    x = args[0] if args else kwargs.get("x", 0.0)
    return backend_module.array(sc.k1(x))


@numpy_eager_registry.register("BesselK1e")
def _np_bessel_k1e(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the bessel k1e logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    import scipy.special as sc

    x = args[0] if args else kwargs.get("x", 0.0)
    return backend_module.array(sc.k1e(x))


@numpy_eager_registry.register("BesselY0")
def _np_bessel_y0(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the bessel y0 logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    import scipy.special as sc

    x = args[0] if args else kwargs.get("x", 0.0)
    return backend_module.array(sc.y0(x))


@numpy_eager_registry.register("BesselY1")
def _np_bessel_y1(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the bessel y1 logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    import scipy.special as sc

    x = args[0] if args else kwargs.get("x", 0.0)
    return backend_module.array(sc.y1(x))


@numpy_eager_registry.register("Dawsn")
def _np_dawsn(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the dawsn logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    import scipy.special as sc

    x = args[0] if args else kwargs.get("x", 0.0)
    return backend_module.array(sc.dawsn(x))


@numpy_eager_registry.register("Expint")
def _np_expint(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the expint logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    import scipy.special as sc

    x = args[0] if args else kwargs.get("x", 0.0)
    n = kwargs.get("n", 1)
    if len(args) > 1:
        n = args[1]
    return backend_module.array(sc.expn(n, x))


@numpy_eager_registry.register("FresnelCos")
def _np_fresnel_cos(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the fresnel cos logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    import scipy.special as sc

    x = args[0] if args else kwargs.get("x", 0.0)
    return backend_module.array(sc.fresnel(x)[1])


@numpy_eager_registry.register("FresnelSin")
def _np_fresnel_sin(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the fresnel sin logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    import scipy.special as sc

    x = args[0] if args else kwargs.get("x", 0.0)
    return backend_module.array(sc.fresnel(x)[0])


@numpy_eager_registry.register("Spence")
def _np_spence(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the spence logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    import scipy.special as sc

    x = args[0] if args else kwargs.get("x", 0.0)
    return backend_module.array(sc.spence(x))


@numpy_eager_registry.register("BesselI0")
def _np_bessel_i0(backend_module: object, x: object, **kwargs: object) -> object:
    """Evaluate the bessel i0 logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        x (object): Required parameter for x.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    import scipy.special as sc

    return backend_module.array(sc.i0(x))


@numpy_eager_registry.register("BesselI1")
def _np_bessel_i1(backend_module: object, x: object, **kwargs: object) -> object:
    """Evaluate the bessel i1 logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        x (object): Required parameter for x.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    import scipy.special as sc

    return backend_module.array(sc.i1(x))


@numpy_eager_registry.register("BesselJn")
def _np_bessel_jn(backend_module: object, x: object, y: object, **kwargs: object) -> object:
    """Evaluate the bessel jn logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        x (object): Required parameter for x.
        y (object): Required parameter for y.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    import scipy.special as sc

    return backend_module.array(sc.jv(x, y))


@numpy_eager_registry.register("Bartlett")
def _np_bartlett(backend_module: object, M: object, **kwargs: object) -> object:
    """Evaluate the bartlett logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        M (object): Required parameter for M.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return np.bartlett(M)
