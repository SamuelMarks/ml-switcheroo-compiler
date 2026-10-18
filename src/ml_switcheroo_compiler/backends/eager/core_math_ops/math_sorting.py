"""math_sorting module."""

from __future__ import annotations

import builtins  # noqa: F401
from typing import cast

from ml_switcheroo_compiler.backends.eager.utils import host_backend
from ml_switcheroo_compiler.backends.eager_registry import EagerValue, global_eager_registry


@global_eager_registry.register("Argpartition")
def _argpartition(backend_module: EagerValue, a: EagerValue, kth: EagerValue, axis: int = -1, **kwargs: EagerValue) -> EagerValue:
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
def _median(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager median operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return cast(EagerValue, getattr(backend_module, "median")(*args, **kwargs))


@global_eager_registry.register("Percentile")
def _percentile(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager percentile operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return cast(EagerValue, getattr(backend_module, "percentile")(*args, **kwargs))


@global_eager_registry.register("Quantile")
def _quantile(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager quantile operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return cast(EagerValue, getattr(backend_module, "quantile")(*args, **kwargs))


@global_eager_registry.register("Partition")
def _np_partition(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
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
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, host_backend.partition(*args, **kwargs))
