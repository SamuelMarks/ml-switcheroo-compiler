"""Implementation of math trig operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.eager.core_math_ops.math_trig`)."""

from __future__ import annotations

import builtins  # noqa: F401

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import EagerValue, ensure_eager_tensor, global_eager_registry
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, get_backend_member


@global_eager_registry.register("Acos")
def _acos(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager acos operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = get_backend_member(backend_module, "arccos", get_backend_member(backend_module, "acos", None))
    if func is not None:
        return ensure_eager_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).arccos(*args, **kwargs)


@global_eager_registry.register("Acosh")
def _acosh(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager acosh operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = get_backend_member(backend_module, "arccosh", get_backend_member(backend_module, "acosh", None))
    if func is not None:
        return ensure_eager_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).arccos(*args, **kwargs)


@global_eager_registry.register("Asin")
def _asin(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager asin operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = get_backend_member(backend_module, "arcsin", get_backend_member(backend_module, "asin", None))
    if func is not None:
        return ensure_eager_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).arcsin(*args, **kwargs)


@global_eager_registry.register("Asinh")
def _asinh(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager asinh operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = get_backend_member(backend_module, "arcsinh", get_backend_member(backend_module, "asinh", None))
    if func is not None:
        return ensure_eager_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).arcsin(*args, **kwargs)


@global_eager_registry.register("Atan")
def _atan(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager atan operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = get_backend_member(backend_module, "arctan", get_backend_member(backend_module, "atan", None))
    if func is not None:
        return ensure_eager_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).arctan(*args, **kwargs)


@global_eager_registry.register("Atanh")
def _atanh(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager atanh operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = get_backend_member(backend_module, "arctanh", get_backend_member(backend_module, "atanh", None))
    if func is not None:
        return ensure_eager_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).arctan(*args, **kwargs)


@global_eager_registry.register("Atan2")
def _atan2(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager atan2 operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = get_backend_member(backend_module, "arctan2", get_backend_member(backend_module, "atan2", None))
    if func is not None:
        return ensure_eager_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).arctan2(*args, **kwargs)


@global_eager_registry.register("Sinc")
def _sinc(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager sinc operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = get_backend_member(backend_module, "sinc", None)
    if func is not None:
        return ensure_eager_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).sinc(*args, **kwargs)


@global_eager_registry.register("Isin")
def _isin(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager isin operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return ensure_eager_tensor(get_backend_member(backend_module, "isin")(*args, **kwargs))


@global_eager_registry.register("Isinf")
def _isinf(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager isinf operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = get_backend_member(backend_module, "isinf", None)
    if func:
        return ensure_eager_tensor(func(*args, **kwargs))

    return _resolve_eager_mod(backend_module).isinf(args[0])


@global_eager_registry.register("Isposinf")
def _isposinf(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager isposinf operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = get_backend_member(backend_module, "isposinf", None)
    if func:
        return ensure_eager_tensor(func(*args, **kwargs))

    return _resolve_eager_mod(backend_module).isposinf(args[0])
