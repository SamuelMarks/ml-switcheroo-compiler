"""Numpy Variable Ops."""

from types import ModuleType
from typing import cast

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, numpy_eager_registry


@numpy_eager_registry.register("Assign")
def _np_assign(backend_module: ModuleType, x: EagerValue, y: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np assign operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        x (EagerValue): The x parameter.
        y (EagerValue): The y parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return y


@numpy_eager_registry.register("Cast")
def _np_cast(backend_module: ModuleType, x: EagerValue, dtype: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np cast operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        x (EagerValue): The x parameter.
        dtype (EagerValue): The dtype parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    dt = getattr(dtype, "value", dtype)
    if isinstance(dt, str):
        if "bfloat" in dt or "float8" in dt:
            dt = "float32"
        elif "int4" in dt:
            dt = "int8"
    return cast(EagerValue, backend_module.asarray(x).astype(dt))


@numpy_eager_registry.register("Bitcast")
def _np_bitcast(backend_module: ModuleType, x: EagerValue, dtype: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np bitcast operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        x (EagerValue): The x parameter.
        dtype (EagerValue): The dtype parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, backend_module.asarray(x).view(getattr(dtype, "value", dtype)))


@numpy_eager_registry.register("ReadVariable")
def _np_read_variable(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np read variable operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return args[0] if args else None


@numpy_eager_registry.register("AssignVariable")
def _np_assign_variable(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np assign variable operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    if len(args) > 1:
        return args[1]
    return None
