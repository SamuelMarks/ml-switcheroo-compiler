"""Numpy Variable Ops."""

# ruff: noqa: E501
from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("Assign")
def _np_assign(backend_module: object, x: object, y: object, *args: object, **kwargs: object) -> object:
    """Evaluate the assign logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        x (object): Required parameter for x.
        y (object): Required parameter for y.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return y


@numpy_eager_registry.register("Cast")
def _np_cast(backend_module: object, x: object, dtype: object, *args: object, **kwargs: object) -> object:
    """Evaluate the cast logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        x (object): Required parameter for x.
        dtype (object): Required parameter for dtype.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    dt = getattr(dtype, "value", dtype)
    if isinstance(dt, str):
        if "bfloat" in dt or "float8" in dt:
            dt = "float32"
        elif "int4" in dt:
            dt = "int8"
    return backend_module.asarray(x).astype(dt)


@numpy_eager_registry.register("Bitcast")
def _np_bitcast(backend_module: object, x: object, dtype: object, *args: object, **kwargs: object) -> object:
    """Evaluate the bitcast logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        x (object): Required parameter for x.
        dtype (object): Required parameter for dtype.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.asarray(x).view(getattr(dtype, "value", dtype))


@numpy_eager_registry.register("ReadVariable")
def _np_read_variable(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the read variable logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return args[0] if args else None


@numpy_eager_registry.register("AssignVariable")
def _np_assign_variable(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the assign variable logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    if len(args) > 1:
        return args[1]
    return None
