"""Numpy Variable Ops."""

from typing import Any

# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("Assign")
def _np_assign(backend_module: Any, x: Any, y: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_assign operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        y (object): The y parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return y


@numpy_eager_registry.register("Cast")
def _np_cast(backend_module: Any, x: Any, dtype: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_cast operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        dtype (object): The dtype parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    dt = getattr(dtype, "value", dtype)
    if isinstance(dt, str):
        if "bfloat" in dt or "float8" in dt:
            dt = "float32"
        elif "int4" in dt:
            dt = "int8"
    return backend_module.asarray(x).astype(dt)


@numpy_eager_registry.register("Bitcast")
def _np_bitcast(backend_module: Any, x: Any, dtype: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_bitcast operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        dtype (object): The dtype parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.asarray(x).view(getattr(dtype, "value", dtype))


@numpy_eager_registry.register("ReadVariable")
def _np_read_variable(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_read_variable operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return args[0] if args else None


@numpy_eager_registry.register("AssignVariable")
def _np_assign_variable(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_assign_variable operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    if len(args) > 1:
        return args[1]
    return None
