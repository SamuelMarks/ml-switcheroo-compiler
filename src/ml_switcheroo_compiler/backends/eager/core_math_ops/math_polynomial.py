"""Implementation of math polynomial operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.eager.core_math_ops.math_polynomial`)."""

from __future__ import annotations

import builtins  # noqa: F401

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import EagerValue, ensure_eager_tensor, global_eager_registry
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, get_backend_member


@global_eager_registry.register("Polyval")
def _polyval(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager polyval operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    func = get_backend_member(backend_module, "polyval", None)
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Polyint")
def _np_polyint(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np polyint operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    func = get_backend_member(backend_module, "polyint", get_backend_member(backend_module, "polyint", None))
    if func is not None:
        return ensure_eager_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).polyint(*args, **kwargs)
