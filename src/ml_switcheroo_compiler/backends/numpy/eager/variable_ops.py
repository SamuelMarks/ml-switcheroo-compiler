"""Numpy Variable Ops."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, numpy_eager_registry
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, get_backend_member


@numpy_eager_registry.register("Assign")
def _np_assign(backend_module: BackendModuleProtocol, x: EagerValue, y: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np assign operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        x (EagerValue): Input `x` argument (x).
        y (EagerValue): Input `y` argument (y).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return y


@numpy_eager_registry.register("Cast")
def _np_cast(backend_module: BackendModuleProtocol, x: EagerValue, dtype: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np cast operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        x (EagerValue): Input `x` argument (x).
        dtype (EagerValue): Input `dtype` argument (dtype).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    dt = get_backend_member(dtype, "value", dtype)
    if isinstance(dt, str):
        if "bfloat" in dt or "float8" in dt:
            dt = "float32"
        elif "int4" in dt:
            dt = "int8"
    return backend_module.asarray(x).astype(dt)


@numpy_eager_registry.register("Bitcast")
def _np_bitcast(backend_module: BackendModuleProtocol, x: EagerValue, dtype: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np bitcast operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        x (EagerValue): Input `x` argument (x).
        dtype (EagerValue): Input `dtype` argument (dtype).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return backend_module.asarray(x).view(get_backend_member(dtype, "value", dtype))


@numpy_eager_registry.register("ReadVariable")
def _np_read_variable(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np read variable operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return args[0] if args else None


@numpy_eager_registry.register("AssignVariable")
def _np_assign_variable(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np assign variable operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    if len(args) > 1:
        return args[1]
    return None
