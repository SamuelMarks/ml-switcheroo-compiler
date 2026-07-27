"""Numpy Logical Reductions."""

# ruff: noqa: E501
from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("All")
def _np_all(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the all logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.all(*args, **kwargs)


@numpy_eager_registry.register("CountNonzero")
def _np_count_nonzero(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the count nonzero logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.count_nonzero(*args, **kwargs)
