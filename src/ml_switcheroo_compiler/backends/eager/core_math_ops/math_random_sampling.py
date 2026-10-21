"""Implementation of math random sampling operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.eager.core_math_ops.math_random_sampling`)."""

from __future__ import annotations

import builtins  # noqa: F401

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import EagerValue, as_tensor, global_eager_registry
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol


@global_eager_registry.register("Pshuffle")
def _pshuffle(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager pshuffle operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    if hasattr(backend_module, "lax") and hasattr(getattr(backend_module, "lax"), "pshuffle"):
        return as_tensor(getattr(backend_module, "lax").pshuffle(*args, **kwargs))
    return args[0] if args else None


@global_eager_registry.register("Gumbel")
def _np_gumbel(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np gumbel operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "gumbel", getattr(backend_module, "gumbel", None))
    if func is not None:
        return as_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).random.gumbel(*args, **kwargs)


@global_eager_registry.register("RngUniform")
def _np_rnguniform(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np rnguniform operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "rnguniform", getattr(backend_module, "rnguniform", None))
    if func is not None:
        return as_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).random.uniform(*args, **kwargs)


@global_eager_registry.register("Wald")
def _np_wald(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np wald operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "wald", getattr(backend_module, "wald", None))
    if func is not None:
        return as_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).random.wald(*args, **kwargs)
