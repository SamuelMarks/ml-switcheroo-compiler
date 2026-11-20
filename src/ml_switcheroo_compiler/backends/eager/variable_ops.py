"""Generic backend-parametric eager reference operations for `variable_ops`."""

from __future__ import annotations

import types

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module


@global_eager_registry.register("Assign")
def _np_assign(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    x: BackendArg,
    y: BackendArg,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager np assign operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        x (BackendArg): Input `x` argument (x).
        y (BackendArg): Input `y` argument (y).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return y


@global_eager_registry.register("Cast")
def _np_cast(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    x: BackendArg,
    dtype: BackendArg,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager np cast operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        x (BackendArg): Input `x` argument (x).
        dtype (BackendArg): Input `dtype` argument (dtype).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    dt = adapt_backend_module(dtype).get_symbol("value", dtype)
    if isinstance(dt, str):
        if "bfloat" in dt or "float8" in dt:
            dt = "float32"
        elif "int4" in dt:
            dt = "int8"
    return adapt_backend_module(backend_module).require_symbol("asarray")(x).astype(dt)


@global_eager_registry.register("Bitcast")
def _np_bitcast(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    x: BackendArg,
    dtype: BackendArg,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager np bitcast operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        x (BackendArg): Input `x` argument (x).
        dtype (BackendArg): Input `dtype` argument (dtype).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return (
        adapt_backend_module(backend_module)
        .require_symbol("asarray")(x)
        .view(adapt_backend_module(dtype).get_symbol("value", dtype))
    )


@global_eager_registry.register("ReadVariable")
def _np_read_variable(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np read variable operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return args[0] if args else None


@global_eager_registry.register("AssignVariable")
def _np_assign_variable(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np assign variable operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    if len(args) > 1:
        return args[1]
    return None
