# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Core utilities."""

from __future__ import annotations

from typing import Any

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry


@global_eager_registry.register("Expm1")
def _expm1(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _expm1 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    x = args[0]
    if hasattr(backend_module, "expm1"):
        return backend_module.expm1(x)
    return backend_module.exp(x) - 1.0


@global_eager_registry.register("FloatPower")
def _float_power(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _float_power operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "float_power", getattr(backend_module, "power", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Frexp")
def _frexp(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _frexp operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    import math

    func = getattr(backend_module, "frexp", getattr(math, "frexp", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Ldexp")
def _ldexp(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _ldexp operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    import math

    func = getattr(backend_module, "ldexp", getattr(math, "ldexp", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Slogdet")
def _slogdet(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _slogdet operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "slogdet"):
        return func.slogdet(*args, **kwargs)
    if hasattr(backend_module, "slogdet"):
        return backend_module.slogdet(*args, **kwargs)

    x = args[0]
    return backend_module.linalg.slogdet(backend_module.asarray(x))


@global_eager_registry.register("Xlog1py")
def _np_xlog1py(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_xlog1py operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "xlog1py", getattr(backend_module, "xlog1py", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return np.where(args[0] == 0, 0, args[0] * np.log1p(args[1]))


@global_eager_registry.register("Xlogy")
def _np_xlogy(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_xlogy operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "xlogy", getattr(backend_module, "xlogy", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return np.where(args[0] == 0, 0, args[0] * np.log(args[1]))
