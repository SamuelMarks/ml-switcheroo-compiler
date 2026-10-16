"""math_creation module."""

from __future__ import annotations

import builtins  # noqa: F401
import typing  # noqa: F401
from typing import cast

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, global_eager_registry
from ml_switcheroo_compiler.distributed.host_collectives import np


@global_eager_registry.register("Fromfunction")
def _fromfunction(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _fromfunction operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return cast(EagerValue, getattr(backend_module, "fromfunction")(*args, **kwargs))


@global_eager_registry.register("FromDlpack")
def _from_dlpack(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _from_dlpack operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return cast(EagerValue, getattr(backend_module, "from_dlpack")(*args, **kwargs))


@global_eager_registry.register("Fromiter")
def _fromiter(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _fromiter operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return cast(EagerValue, getattr(backend_module, "fromiter")(*args, **kwargs))


@global_eager_registry.register("Frompyfunc")
def _frompyfunc(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _frompyfunc operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return cast(EagerValue, getattr(backend_module, "frompyfunc")(*args, **kwargs))


@global_eager_registry.register("Geomspace")
def _geomspace(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _geomspace operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return cast(EagerValue, getattr(backend_module, "geomspace")(*args, **kwargs))


@global_eager_registry.register("Mgrid")
def _mgrid(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _mgrid operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return cast(EagerValue, getattr(backend_module, "mgrid")(*args, **kwargs))


@global_eager_registry.register("Ogrid")
def _ogrid(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _ogrid operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return cast(EagerValue, getattr(backend_module, "ogrid")(*args, **kwargs))


@global_eager_registry.register("NpFromfunction")
def _np_fromfunction(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_fromfunction operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    func = getattr(backend_module, "fromfunction", None)
    if func:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, np.fromfunction(*args, **kwargs))


@global_eager_registry.register("NpFromiter")
def _np_fromiter(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_fromiter operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    func = getattr(backend_module, "fromiter", None)
    if func:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, np.fromiter(*args, **kwargs))


@global_eager_registry.register("NpFrompyfunc")
def _np_frompyfunc(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_frompyfunc operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    func = getattr(backend_module, "frompyfunc", None)
    if func:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, np.frompyfunc(*args, **kwargs))
