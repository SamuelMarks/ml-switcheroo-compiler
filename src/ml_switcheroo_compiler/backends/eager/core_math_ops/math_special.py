"""Core utilities."""

from __future__ import annotations

import builtins  # noqa: F401
from types import ModuleType
from typing import cast

from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue, global_eager_registry
from ml_switcheroo_compiler.distributed.host_collectives import _scipy_mod


@global_eager_registry.register("Erf")
def _erf(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _erf operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    mod = cast(ModuleType, backend_module)
    x = cast(EagerTensorProtocol, args[0])
    if hasattr(mod, "erf"):
        return cast(EagerValue, mod.erf(x))
    # A&S approximation 7.1.26
    # erf(x) = 1 - (a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5)*e^{-x^2}
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429

    sign: EagerTensorProtocol = mod.sign(x)
    abs_x: EagerTensorProtocol = mod.abs(x)
    t: EagerTensorProtocol = 1.0 / (1.0 + p * abs_x)
    y: EagerTensorProtocol = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * mod.exp(-abs_x * abs_x)
    return sign * y


@global_eager_registry.register("Erfc")
def _erfc(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _erfc operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    mod = cast(ModuleType, backend_module)
    x = cast(EagerTensorProtocol, args[0])
    if hasattr(mod, "erfc"):
        return cast(EagerValue, mod.erfc(x))
    return 1.0 - cast(EagerTensorProtocol, _erf(backend_module, x))


@global_eager_registry.register("Erfinv")
def _erfinv(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _erfinv operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    mod = cast(ModuleType, backend_module)
    x = cast(EagerTensorProtocol, args[0])
    if hasattr(mod, "erfinv"):
        return cast(EagerValue, mod.erfinv(x))
    # Approximation of erfinv
    # https://en.wikipedia.org/wiki/Error_function#Approximation_with_elementary_functions
    # For a naive approximation:
    a = 0.147
    ln_1_x2: EagerTensorProtocol = mod.log(1.0 - x * x + 1e-12)
    term1: EagerTensorProtocol = 2.0 / (3.141592653589793 * a) + ln_1_x2 / 2.0
    term2: EagerTensorProtocol = ln_1_x2 / a
    return cast(EagerValue, mod.sign(x) * mod.sqrt(mod.sqrt(term1 * term1 - term2) - term1))


@global_eager_registry.register("BesselI0")
def _bessel_i0(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _bessel_i0 operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return cast(EagerValue, _scipy_mod("special").i0(*args, **kwargs))


@global_eager_registry.register("BesselI0e")
def _bessel_i0e(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _bessel_i0e operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return cast(EagerValue, _scipy_mod("special").i0e(*args, **kwargs))


@global_eager_registry.register("BesselI1")
def _bessel_i1(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _bessel_i1 operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return cast(EagerValue, _scipy_mod("special").i1(*args, **kwargs))


@global_eager_registry.register("BesselI1e")
def _bessel_i1e(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _bessel_i1e operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return cast(EagerValue, _scipy_mod("special").i1e(*args, **kwargs))


@global_eager_registry.register("BesselJ0")
def _bessel_j0(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _bessel_j0 operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return cast(EagerValue, _scipy_mod("special").j0(*args, **kwargs))


@global_eager_registry.register("BesselJ1")
def _bessel_j1(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _bessel_j1 operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return cast(EagerValue, _scipy_mod("special").j1(*args, **kwargs))


@global_eager_registry.register("BesselJn")
def _bessel_jn(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _bessel_jn operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return cast(EagerValue, _scipy_mod("special").jv(*args, **kwargs))


@global_eager_registry.register("BesselK0")
def _bessel_k0(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _bessel_k0 operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return cast(EagerValue, _scipy_mod("special").k0(*args, **kwargs))


@global_eager_registry.register("BesselK0e")
def _bessel_k0e(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _bessel_k0e operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return cast(EagerValue, _scipy_mod("special").k0e(*args, **kwargs))


@global_eager_registry.register("BesselK1")
def _bessel_k1(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _bessel_k1 operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return cast(EagerValue, _scipy_mod("special").k1(*args, **kwargs))


@global_eager_registry.register("BesselK1e")
def _bessel_k1e(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _bessel_k1e operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return cast(EagerValue, _scipy_mod("special").k1e(*args, **kwargs))


@global_eager_registry.register("BesselY0")
def _bessel_y0(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _bessel_y0 operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return cast(EagerValue, _scipy_mod("special").y0(*args, **kwargs))


@global_eager_registry.register("BesselY1")
def _bessel_y1(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _bessel_y1 operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return cast(EagerValue, _scipy_mod("special").y1(*args, **kwargs))


@global_eager_registry.register("Digamma")
def _digamma(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _digamma operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return cast(EagerValue, _scipy_mod("special").digamma(*args, **kwargs))


@global_eager_registry.register("Igammac")
def _igammac(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _igammac operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return cast(EagerValue, _scipy_mod("special").gammaincc(*args, **kwargs))


@global_eager_registry.register("Polygamma")
def _polygamma(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _polygamma operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return cast(EagerValue, _scipy_mod("special").polygamma(*args, **kwargs))


@global_eager_registry.register("Igamma")
def _igamma(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _igamma operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return cast(EagerValue, _scipy_mod("special").gammainc(*args, **kwargs))


@global_eager_registry.register("Gamma")
def _gamma(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _gamma operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return cast(EagerValue, _scipy_mod("special").gamma(*args, **kwargs))


@global_eager_registry.register("ModifiedBesselI1")
def _np_modifiedbesseli1(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_modifiedbesseli1 operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return cast(EagerValue, _scipy_mod("special").i1(*args, **kwargs))
