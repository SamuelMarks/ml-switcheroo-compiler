"""Implementation of math creation operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.eager.core_math_ops.math_creation`)."""

from __future__ import annotations

import builtins  # noqa: F401
import typing  # noqa: F401

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import EagerValue, as_tensor, global_eager_registry
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol


@global_eager_registry.register("Fromfunction")
def _fromfunction(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager fromfunction operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return as_tensor(getattr(backend_module, "fromfunction")(*args, **kwargs))


@global_eager_registry.register("FromDlpack")
def _from_dlpack(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager from dlpack operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return as_tensor(getattr(backend_module, "from_dlpack")(*args, **kwargs))


@global_eager_registry.register("Fromiter")
def _fromiter(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager fromiter operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return as_tensor(getattr(backend_module, "fromiter")(*args, **kwargs))


@global_eager_registry.register("Frompyfunc")
def _frompyfunc(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager frompyfunc operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return as_tensor(getattr(backend_module, "frompyfunc")(*args, **kwargs))


@global_eager_registry.register("Geomspace")
def _geomspace(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager geomspace operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return as_tensor(getattr(backend_module, "geomspace")(*args, **kwargs))


@global_eager_registry.register("Mgrid")
def _mgrid(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager mgrid operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return as_tensor(getattr(backend_module, "mgrid")(*args, **kwargs))


@global_eager_registry.register("Ogrid")
def _ogrid(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager ogrid operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return as_tensor(getattr(backend_module, "ogrid")(*args, **kwargs))


@global_eager_registry.register("NpFromfunction")
def _np_fromfunction(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np fromfunction operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "fromfunction", None)
    if func:
        return as_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).fromfunction(*args, **kwargs)


@global_eager_registry.register("NpFromiter")
def _np_fromiter(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np fromiter operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "fromiter", None)
    if func:
        return as_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).fromiter(*args, **kwargs)


@global_eager_registry.register("NpFrompyfunc")
def _np_frompyfunc(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np frompyfunc operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "frompyfunc", None)
    if func:
        return as_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).frompyfunc(*args, **kwargs)
