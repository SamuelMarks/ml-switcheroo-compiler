"""Core utilities."""

from __future__ import annotations

import builtins  # noqa: F401
from typing import cast

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, global_eager_registry
from ml_switcheroo_compiler.distributed.host_collectives import np


@global_eager_registry.register("Expm1")
def _expm1(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _expm1 operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    x = args[0]
    if hasattr(backend_module, "expm1"):
        return cast(EagerValue, getattr(backend_module, "expm1")(x))
    return cast(EagerValue, getattr(backend_module, "exp")(x) - 1.0)


@global_eager_registry.register("FloatPower")
def _float_power(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _float_power operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    func = getattr(backend_module, "float_power", getattr(backend_module, "power", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Frexp")
def _frexp(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _frexp operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    import math

    func = getattr(backend_module, "frexp", getattr(math, "frexp", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Ldexp")
def _ldexp(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _ldexp operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    import math

    func = getattr(backend_module, "ldexp", getattr(math, "ldexp", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Slogdet")
def _slogdet(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _slogdet operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "slogdet"):
        return cast(EagerValue, func.slogdet(*args, **kwargs))
    if hasattr(backend_module, "slogdet"):
        return cast(EagerValue, getattr(backend_module, "slogdet")(*args, **kwargs))

    x = args[0]
    return cast(EagerValue, getattr(backend_module, "linalg").slogdet(getattr(backend_module, "asarray")(x)))


@global_eager_registry.register("Xlog1py")
def _np_xlog1py(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_xlog1py operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    func = getattr(backend_module, "xlog1py", getattr(backend_module, "xlog1py", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, np.where(args[0] == 0, 0, args[0] * np.log1p(args[1])))


@global_eager_registry.register("Xlogy")
def _np_xlogy(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_xlogy operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    func = getattr(backend_module, "xlogy", getattr(backend_module, "xlogy", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, np.where(args[0] == 0, 0, args[0] * np.log(args[1])))
