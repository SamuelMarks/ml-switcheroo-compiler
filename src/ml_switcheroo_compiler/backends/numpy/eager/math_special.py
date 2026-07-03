"""Numpy eager special math functions."""

import numpy as np
import scipy.special
import scipy.special as sc

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("BesselJ0")
def _np_bessel_j0(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring."""
    return scipy.special.j0(*args, **kwargs)


@numpy_eager_registry.register("BesselJ1")
def _np_bessel_j1(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring."""
    return scipy.special.j1(*args, **kwargs)


@numpy_eager_registry.register("BesselK0")
def _np_bessel_k0(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring."""
    return scipy.special.k0(*args, **kwargs)


@numpy_eager_registry.register("BesselK0e")
def _np_bessel_k0e(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring."""
    return scipy.special.k0e(*args, **kwargs)


@numpy_eager_registry.register("BesselK1")
def _np_bessel_k1(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring."""
    return scipy.special.k1(*args, **kwargs)


@numpy_eager_registry.register("BesselK1e")
def _np_bessel_k1e(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring."""
    return scipy.special.k1e(*args, **kwargs)


@numpy_eager_registry.register("BesselY0")
def _np_bessel_y0(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring."""
    return scipy.special.y0(*args, **kwargs)


@numpy_eager_registry.register("BesselY1")
def _np_bessel_y1(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring."""
    return scipy.special.y1(*args, **kwargs)


@numpy_eager_registry.register("Dawsn")
def _np_dawsn(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring."""
    return scipy.special.dawsn(*args, **kwargs)


@numpy_eager_registry.register("Expint")
def _np_expint(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring."""
    return scipy.special.expi(*args, **kwargs)


@numpy_eager_registry.register("FresnelCos")
def _np_fresnel_cos(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring."""
    return scipy.special.fresnel(*args, **kwargs)[1]


@numpy_eager_registry.register("FresnelSin")
def _np_fresnel_sin(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring."""
    return scipy.special.fresnel(*args, **kwargs)[0]


@numpy_eager_registry.register("Spence")
def _np_spence(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring."""
    return scipy.special.spence(*args, **kwargs)


@numpy_eager_registry.register("BesselI0")
def _np_bessel_i0(backend_module: object, x: object, **kwargs: object) -> object:
    """Function docstring."""
    return sc.i0(x)


@numpy_eager_registry.register("BesselI1")
def _np_bessel_i1(backend_module: object, x: object, **kwargs: object) -> object:
    """Function docstring."""
    return sc.i1(x)


@numpy_eager_registry.register("BesselJn")
def _np_bessel_jn(backend_module: object, x: object, y: object, **kwargs: object) -> object:
    """Function docstring."""
    return sc.jn(x, y)


@numpy_eager_registry.register("Bartlett")
def _np_bartlett(backend_module: object, M: object, **kwargs: object) -> object:
    """Function docstring."""
    return np.bartlett(M)
