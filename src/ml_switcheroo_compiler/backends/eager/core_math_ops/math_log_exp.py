"""Implementation of math log exp operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.eager.core_math_ops.math_log_exp`)."""

from __future__ import annotations

import builtins  # noqa: F401

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import EagerValue, as_tensor, global_eager_registry
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol


@global_eager_registry.register("Expm1")
def _expm1(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager expm1 operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    x = args[0]
    if hasattr(backend_module, "expm1"):
        return as_tensor(getattr(backend_module, "expm1")(x))
    return as_tensor(getattr(backend_module, "exp")(x) - 1.0)


@global_eager_registry.register("FloatPower")
def _float_power(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager float power operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "float_power", getattr(backend_module, "power", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Frexp")
def _frexp(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager frexp operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    import math

    func = getattr(backend_module, "frexp", getattr(math, "frexp", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Ldexp")
def _ldexp(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager ldexp operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    import math

    func = getattr(backend_module, "ldexp", getattr(math, "ldexp", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Slogdet")
def _slogdet(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager slogdet operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "slogdet"):
        return as_tensor(func.slogdet(*args, **kwargs))
    if hasattr(backend_module, "slogdet"):
        return as_tensor(getattr(backend_module, "slogdet")(*args, **kwargs))

    x = args[0]
    return as_tensor(getattr(backend_module, "linalg").slogdet(getattr(backend_module, "asarray")(x)))


@global_eager_registry.register("Xlog1py")
def _np_xlog1py(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np xlog1py operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "xlog1py", getattr(backend_module, "xlog1py", None))
    if func is not None:
        return as_tensor(func(*args, **kwargs))
    mod = _resolve_eager_mod(backend_module)
    return mod.where(args[0] == 0, 0, args[0] * mod.log1p(args[1]))


@global_eager_registry.register("Xlogy")
def _np_xlogy(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np xlogy operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "xlogy", getattr(backend_module, "xlogy", None))
    if func is not None:
        return as_tensor(func(*args, **kwargs))
    mod = _resolve_eager_mod(backend_module)
    return mod.where(args[0] == 0, 0, args[0] * mod.log(args[1]))
