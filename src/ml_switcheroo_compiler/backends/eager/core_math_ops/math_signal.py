"""Operations and utilities for `ml_switcheroo_compiler.backends.eager.core_math_ops.math_signal`."""

from __future__ import annotations

import builtins  # noqa: F401
import types
from collections.abc import Callable  # noqa: F401

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module


@global_eager_registry.register("Correlate")
def _correlate(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager correlate operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("correlate", None)
    if func is not None:
        return func(*args, **kwargs)
    (a, v) = (args[0], args[1])
    mode = kwargs.get("mode", "valid") if adapt_backend_module(kwargs).has_symbol("get") else "valid"
    mod = _resolve_eager_mod(backend_module)
    return mod.correlate(mod.asarray(a), mod.asarray(v), mode=mode)


@global_eager_registry.register("WindowHann")
def _np_windowhann(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np windowhann operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "windowhann", adapt_backend_module(backend_module).get_symbol("windowhann", None)
    )
    if func is not None:
        return func(*args, **kwargs)
    return _resolve_eager_mod(backend_module).hanning(*args, **kwargs)
