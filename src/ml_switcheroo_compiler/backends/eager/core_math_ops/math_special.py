"""Core special mathematical functions and numerical approximations for eager backends."""

from __future__ import annotations

import builtins  # noqa: F401
import types

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    EagerTensorProtocol,
    global_eager_registry,
    numpy_eager_registry,
    require_callable,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module


def _dispatch_special(
    op_name: str,
    attr_name: str,
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Dispatch a special mathematical function to `backend_module` or `numpy_eager_registry`.

    Args:
        op_name: Registered operation name in `numpy_eager_registry`.
        attr_name: Attribute name on `backend_module` or `backend_module.special`.
        backend_module: Active backend module.
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        BackendArg: Evaluated special function result.
    """
    special_mod = adapt_backend_module(backend_module).get_symbol("special", backend_module)
    func = adapt_backend_module(special_mod).get_symbol(
        attr_name, adapt_backend_module(backend_module).get_symbol(attr_name, None)
    )
    if callable(func):
        return func(*args, **kwargs)
    from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod

    np_handler = numpy_eager_registry.get(op_name)
    return require_callable(np_handler)(require_callable(_resolve_eager_mod)(backend_module), *args, **kwargs)


@global_eager_registry.register("Erf")
def _erf(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager erf operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    mod = backend_module
    x = args[0]
    if adapt_backend_module(mod).has_symbol("erf"):
        _typed_ret_1: BackendArg = adapt_backend_module(mod).require_symbol("erf")(x)
        return _typed_ret_1
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429

    sign: EagerTensorProtocol = adapt_backend_module(mod).require_symbol("sign")(x)
    abs_x: EagerTensorProtocol = adapt_backend_module(mod).require_symbol("abs")(x)
    t: EagerTensorProtocol = 1.0 / (1.0 + p * abs_x)
    y: EagerTensorProtocol = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * adapt_backend_module(
        mod
    ).require_symbol("exp")(-abs_x * abs_x)
    return sign * y


@global_eager_registry.register("Erfc")
def _erfc(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager erfc operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    mod = backend_module
    x = args[0]
    if adapt_backend_module(mod).has_symbol("erfc"):
        _typed_ret_2: BackendArg = adapt_backend_module(mod).require_symbol("erfc")(x)
        return _typed_ret_2
    from ml_switcheroo_compiler.backends.eager_registry import coerce_numeric

    return 1.0 - coerce_numeric(_erf(backend_module, x))


@global_eager_registry.register("Erfinv")
def _erfinv(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager erfinv operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    from ml_switcheroo_compiler.backends.eager_registry import coerce_numeric

    mod = backend_module
    x = coerce_numeric(args[0])
    if adapt_backend_module(mod).has_symbol("erfinv"):
        _typed_ret_3: BackendArg = adapt_backend_module(mod).require_symbol("erfinv")(x)
        return _typed_ret_3
    a = 0.147
    ln_1_x2: EagerTensorProtocol = adapt_backend_module(mod).require_symbol("log")(1.0 - x * x + 1e-12)
    term1: EagerTensorProtocol = 2.0 / (3.141592653589793 * a) + ln_1_x2 / 2.0
    term2: EagerTensorProtocol = ln_1_x2 / a
    return adapt_backend_module(mod).require_symbol("sign")(x) * adapt_backend_module(mod).require_symbol("sqrt")(
        adapt_backend_module(mod).require_symbol("sqrt")(term1 * term1 - term2) - term1
    )


@global_eager_registry.register("BesselI0")
def _bessel_i0(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager bessel i0 operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return _dispatch_special("BesselI0", "i0", backend_module, *args, **kwargs)


@global_eager_registry.register("BesselI0e")
def _bessel_i0e(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager bessel i0e operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return _dispatch_special("BesselI0e", "i0e", backend_module, *args, **kwargs)


@global_eager_registry.register("BesselI1")
def _bessel_i1(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager bessel i1 operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return _dispatch_special("BesselI1", "i1", backend_module, *args, **kwargs)


@global_eager_registry.register("BesselI1e")
def _bessel_i1e(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager bessel i1e operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return _dispatch_special("BesselI1e", "i1e", backend_module, *args, **kwargs)


@global_eager_registry.register("BesselJ0")
def _bessel_j0(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager bessel j0 operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return _dispatch_special("BesselJ0", "j0", backend_module, *args, **kwargs)


@global_eager_registry.register("BesselJ1")
def _bessel_j1(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager bessel j1 operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return _dispatch_special("BesselJ1", "j1", backend_module, *args, **kwargs)


@global_eager_registry.register("BesselJn")
def _bessel_jn(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager bessel jn operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return _dispatch_special("BesselJn", "jv", backend_module, *args, **kwargs)


@global_eager_registry.register("BesselK0")
def _bessel_k0(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager bessel k0 operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return _dispatch_special("BesselK0", "k0", backend_module, *args, **kwargs)


@global_eager_registry.register("BesselK0e")
def _bessel_k0e(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager bessel k0e operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return _dispatch_special("BesselK0e", "k0e", backend_module, *args, **kwargs)


@global_eager_registry.register("BesselK1")
def _bessel_k1(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager bessel k1 operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return _dispatch_special("BesselK1", "k1", backend_module, *args, **kwargs)


@global_eager_registry.register("BesselK1e")
def _bessel_k1e(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager bessel k1e operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return _dispatch_special("BesselK1e", "k1e", backend_module, *args, **kwargs)


@global_eager_registry.register("BesselY0")
def _bessel_y0(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager bessel y0 operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return _dispatch_special("BesselY0", "y0", backend_module, *args, **kwargs)


@global_eager_registry.register("BesselY1")
def _bessel_y1(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager bessel y1 operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return _dispatch_special("BesselY1", "y1", backend_module, *args, **kwargs)


@global_eager_registry.register("Digamma")
def _digamma(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager digamma operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return _dispatch_special("Digamma", "digamma", backend_module, *args, **kwargs)


@global_eager_registry.register("Igammac")
def _igammac(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager igammac operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return _dispatch_special("Igammac", "gammaincc", backend_module, *args, **kwargs)


@global_eager_registry.register("Polygamma")
def _polygamma(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager polygamma operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return _dispatch_special("Polygamma", "polygamma", backend_module, *args, **kwargs)


@global_eager_registry.register("Igamma")
def _igamma(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager igamma operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return _dispatch_special("Igamma", "gammainc", backend_module, *args, **kwargs)


@global_eager_registry.register("Gamma")
def _gamma(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager gamma operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return _dispatch_special("Gamma", "gamma", backend_module, *args, **kwargs)


@global_eager_registry.register("ModifiedBesselI1")
def _np_modifiedbesseli1(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np modifiedbesseli1 operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return _dispatch_special("ModifiedBesselI1", "i1", backend_module, *args, **kwargs)
