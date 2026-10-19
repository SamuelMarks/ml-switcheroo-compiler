"""Core special mathematical functions and numerical approximations for eager backends."""

from __future__ import annotations

import builtins  # noqa: F401
from collections.abc import Callable
from types import ModuleType
from typing import cast

from ml_switcheroo_compiler.backends.eager_registry import (
    EagerTensorProtocol,
    EagerValue,
    global_eager_registry,
    numpy_eager_registry,
)


def _dispatch_special(op_name: str, attr_name: str, backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Dispatch a special mathematical function to `backend_module` or `numpy_eager_registry`.

    Args:
        op_name: Registered operation name in `numpy_eager_registry`.
        attr_name: Attribute name on `backend_module` or `backend_module.special`.
        backend_module: Active backend module.
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        EagerValue: Evaluated special function result.
    """
    special_mod = getattr(backend_module, "special", backend_module)
    func = getattr(special_mod, attr_name, getattr(backend_module, attr_name, None))
    if callable(func):
        return cast(EagerValue, func(*args, **kwargs))
    from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod

    np_handler = cast("Callable[..., EagerValue]", numpy_eager_registry.get(op_name))
    return np_handler(_resolve_eager_mod(backend_module), *args, **kwargs)


@global_eager_registry.register("Erf")
def _erf(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager erf operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    mod = cast(ModuleType, backend_module)
    x = cast(EagerTensorProtocol, args[0])
    if hasattr(mod, "erf"):
        return cast(EagerValue, mod.erf(x))
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
    """Execute the eager erfc operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    mod = cast(ModuleType, backend_module)
    x = cast(EagerTensorProtocol, args[0])
    if hasattr(mod, "erfc"):
        return cast(EagerValue, mod.erfc(x))
    return 1.0 - cast(EagerTensorProtocol, _erf(backend_module, x))


@global_eager_registry.register("Erfinv")
def _erfinv(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager erfinv operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    mod = cast(ModuleType, backend_module)
    x = cast(EagerTensorProtocol, args[0])
    if hasattr(mod, "erfinv"):
        return cast(EagerValue, mod.erfinv(x))
    a = 0.147
    ln_1_x2: EagerTensorProtocol = mod.log(1.0 - x * x + 1e-12)
    term1: EagerTensorProtocol = 2.0 / (3.141592653589793 * a) + ln_1_x2 / 2.0
    term2: EagerTensorProtocol = ln_1_x2 / a
    return cast(EagerValue, mod.sign(x) * mod.sqrt(mod.sqrt(term1 * term1 - term2) - term1))


@global_eager_registry.register("BesselI0")
def _bessel_i0(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager bessel i0 operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return _dispatch_special("BesselI0", "i0", backend_module, *args, **kwargs)


@global_eager_registry.register("BesselI0e")
def _bessel_i0e(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager bessel i0e operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return _dispatch_special("BesselI0e", "i0e", backend_module, *args, **kwargs)


@global_eager_registry.register("BesselI1")
def _bessel_i1(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager bessel i1 operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return _dispatch_special("BesselI1", "i1", backend_module, *args, **kwargs)


@global_eager_registry.register("BesselI1e")
def _bessel_i1e(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager bessel i1e operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return _dispatch_special("BesselI1e", "i1e", backend_module, *args, **kwargs)


@global_eager_registry.register("BesselJ0")
def _bessel_j0(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager bessel j0 operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return _dispatch_special("BesselJ0", "j0", backend_module, *args, **kwargs)


@global_eager_registry.register("BesselJ1")
def _bessel_j1(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager bessel j1 operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return _dispatch_special("BesselJ1", "j1", backend_module, *args, **kwargs)


@global_eager_registry.register("BesselJn")
def _bessel_jn(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager bessel jn operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return _dispatch_special("BesselJn", "jv", backend_module, *args, **kwargs)


@global_eager_registry.register("BesselK0")
def _bessel_k0(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager bessel k0 operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return _dispatch_special("BesselK0", "k0", backend_module, *args, **kwargs)


@global_eager_registry.register("BesselK0e")
def _bessel_k0e(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager bessel k0e operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return _dispatch_special("BesselK0e", "k0e", backend_module, *args, **kwargs)


@global_eager_registry.register("BesselK1")
def _bessel_k1(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager bessel k1 operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return _dispatch_special("BesselK1", "k1", backend_module, *args, **kwargs)


@global_eager_registry.register("BesselK1e")
def _bessel_k1e(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager bessel k1e operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return _dispatch_special("BesselK1e", "k1e", backend_module, *args, **kwargs)


@global_eager_registry.register("BesselY0")
def _bessel_y0(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager bessel y0 operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return _dispatch_special("BesselY0", "y0", backend_module, *args, **kwargs)


@global_eager_registry.register("BesselY1")
def _bessel_y1(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager bessel y1 operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return _dispatch_special("BesselY1", "y1", backend_module, *args, **kwargs)


@global_eager_registry.register("Digamma")
def _digamma(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager digamma operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return _dispatch_special("Digamma", "digamma", backend_module, *args, **kwargs)


@global_eager_registry.register("Igammac")
def _igammac(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager igammac operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return _dispatch_special("Igammac", "gammaincc", backend_module, *args, **kwargs)


@global_eager_registry.register("Polygamma")
def _polygamma(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager polygamma operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return _dispatch_special("Polygamma", "polygamma", backend_module, *args, **kwargs)


@global_eager_registry.register("Igamma")
def _igamma(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager igamma operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return _dispatch_special("Igamma", "gammainc", backend_module, *args, **kwargs)


@global_eager_registry.register("Gamma")
def _gamma(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager gamma operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return _dispatch_special("Gamma", "gamma", backend_module, *args, **kwargs)


@global_eager_registry.register("ModifiedBesselI1")
def _np_modifiedbesseli1(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np modifiedbesseli1 operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return _dispatch_special("ModifiedBesselI1", "i1", backend_module, *args, **kwargs)
