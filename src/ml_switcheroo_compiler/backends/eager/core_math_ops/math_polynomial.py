"""Implementation of math polynomial operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.eager.core_math_ops.math_polynomial`)."""

from __future__ import annotations

import builtins  # noqa: F401
import typing

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import EagerValue, global_eager_registry
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol


@global_eager_registry.register("Polyval")
def _polyval(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager polyval operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "polyval", None)
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Polyint")
def _np_polyint(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np polyint operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "polyint", getattr(backend_module, "polyint", None))
    if func is not None:
        return typing.cast("EagerValue", func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).polyint(*args, **kwargs)
