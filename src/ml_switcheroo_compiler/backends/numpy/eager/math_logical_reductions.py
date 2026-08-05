"""Numpy Logical Reductions."""

# ruff: noqa: E501
from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("All")
def _np_all(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_all operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return backend_module.all(*args, **kwargs)


@numpy_eager_registry.register("CountNonzero")
def _np_count_nonzero(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_count_nonzero operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return backend_module.count_nonzero(*args, **kwargs)
