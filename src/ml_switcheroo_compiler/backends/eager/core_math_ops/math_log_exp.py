"""Operations and utilities for `ml_switcheroo_compiler.backends.eager.core_math_ops.math_log_exp`."""

from __future__ import annotations

import builtins  # noqa: F401
import types

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    global_eager_registry,
    require_callable,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module


@global_eager_registry.register("Expm1")
def _expm1(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager expm1 operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    x = args[0]
    if adapt_backend_module(backend_module).has_symbol("expm1"):
        return adapt_backend_module(backend_module).require_symbol("expm1")(x)
    return adapt_backend_module(backend_module).require_symbol("exp")(x) - 1.0


@global_eager_registry.register("FloatPower")
def _float_power(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager float power operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "float_power", adapt_backend_module(backend_module).get_symbol("power", None)
    )
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Frexp")
def _frexp(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager frexp operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    import math

    func = adapt_backend_module(backend_module).get_symbol(
        "frexp", adapt_backend_module(math).get_symbol("frexp", None)
    )
    return require_callable(func)(*args, **kwargs) if func else None


@global_eager_registry.register("Ldexp")
def _ldexp(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager ldexp operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    import math

    func = adapt_backend_module(backend_module).get_symbol(
        "ldexp", adapt_backend_module(math).get_symbol("ldexp", None)
    )
    return require_callable(func)(*args, **kwargs) if func else None


@global_eager_registry.register("Slogdet")
def _slogdet(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager slogdet operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("linalg", None)
    if func and adapt_backend_module(func).has_symbol("slogdet"):
        return func.slogdet(*args, **kwargs)
    if adapt_backend_module(backend_module).has_symbol("slogdet"):
        return adapt_backend_module(backend_module).require_symbol("slogdet")(*args, **kwargs)

    x = args[0]
    return (
        adapt_backend_module(backend_module)
        .require_symbol("linalg")
        .slogdet(adapt_backend_module(backend_module).require_symbol("asarray")(x))
    )


@global_eager_registry.register("Xlog1py")
def _np_xlog1py(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np xlog1py operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "xlog1py", adapt_backend_module(backend_module).get_symbol("xlog1py", None)
    )
    if func is not None:
        return func(*args, **kwargs)
    mod = _resolve_eager_mod(backend_module)
    return mod.where(args[0] == 0, 0, args[0] * mod.log1p(args[1]))


@global_eager_registry.register("Xlogy")
def _np_xlogy(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np xlogy operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "xlogy", adapt_backend_module(backend_module).get_symbol("xlogy", None)
    )
    if func is not None:
        return func(*args, **kwargs)
    mod = _resolve_eager_mod(backend_module)
    return mod.where(args[0] == 0, 0, args[0] * mod.log(args[1]))
