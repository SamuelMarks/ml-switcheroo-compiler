"""Implementation of math sorting operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.eager.core_math_ops.math_sorting`)."""

from __future__ import annotations

import builtins  # noqa: F401

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import EagerValue, as_tensor, global_eager_registry
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol


@global_eager_registry.register("Argpartition")
def _argpartition(backend_module: BackendModuleProtocol, a: EagerValue, kth: EagerValue, axis: int = -1, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager argpartition operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        a: The a parameter.
        kth: The kth parameter.
        axis (int): The axis parameter.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return getattr(backend_module, "argsort")(a, axis=axis) if hasattr(backend_module, "argsort") else a


@global_eager_registry.register("Median")
def _median(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager median operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return as_tensor(getattr(backend_module, "median")(*args, **kwargs))


@global_eager_registry.register("Percentile")
def _percentile(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager percentile operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return as_tensor(getattr(backend_module, "percentile")(*args, **kwargs))


@global_eager_registry.register("Quantile")
def _quantile(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager quantile operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return as_tensor(getattr(backend_module, "quantile")(*args, **kwargs))


@global_eager_registry.register("Partition")
def _np_partition(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np partition operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "partition", getattr(backend_module, "partition", None))
    if func is not None:
        return as_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).partition(*args, **kwargs)
