"""Implementation of math signal operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.eager.core_math_ops.math_signal`)."""

from __future__ import annotations

import builtins  # noqa: F401
from collections.abc import Callable  # noqa: F401

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import EagerValue, ensure_eager_tensor, global_eager_registry
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, get_backend_member, has_backend_member


@global_eager_registry.register("Correlate")
def _correlate(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager correlate operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    func = get_backend_member(backend_module, "correlate", None)
    if func is not None:
        return ensure_eager_tensor(func(*args, **kwargs))
    (a, v) = (args[0], args[1])
    mode = kwargs.get("mode", "valid") if has_backend_member(kwargs, "get") else "valid"
    mod = _resolve_eager_mod(backend_module)
    return mod.correlate(mod.asarray(a), mod.asarray(v), mode=mode)


@global_eager_registry.register("WindowHann")
def _np_windowhann(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np windowhann operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    func = get_backend_member(backend_module, "windowhann", get_backend_member(backend_module, "windowhann", None))
    if func is not None:
        return ensure_eager_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).hanning(*args, **kwargs)
