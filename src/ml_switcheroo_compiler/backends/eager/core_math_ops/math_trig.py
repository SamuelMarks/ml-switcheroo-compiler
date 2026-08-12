# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Core utilities."""

from __future__ import annotations

from typing import Any

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry


@global_eager_registry.register("Acos")
def _acos(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _acos operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "arccos", getattr(backend_module, "acos", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Acosh")
def _acosh(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _acosh operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "arccosh", getattr(backend_module, "acosh", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Asin")
def _asin(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _asin operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "arcsin", getattr(backend_module, "asin", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Asinh")
def _asinh(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _asinh operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "arcsinh", getattr(backend_module, "asinh", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Atan")
def _atan(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _atan operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "arctan", getattr(backend_module, "atan", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Atanh")
def _atanh(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _atanh operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "arctanh", getattr(backend_module, "atanh", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Atan2")
def _atan2(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _atan2 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "arctan2", getattr(backend_module, "atan2", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Sinc")
def _sinc(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _sinc operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "sinc", None)
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Isin")
def _isin(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _isin operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.isin(*args, **kwargs)


@global_eager_registry.register("Isinf")
def _isinf(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _isinf operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "isinf", None)
    if func:
        return func(*args, **kwargs)

    return backend_module.isinf(backend_module.asarray(args[0]))


@global_eager_registry.register("Isposinf")
def _isposinf(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _isposinf operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "isposinf", None)
    if func:
        return func(*args, **kwargs)

    return backend_module.isposinf(backend_module.asarray(args[0]))
