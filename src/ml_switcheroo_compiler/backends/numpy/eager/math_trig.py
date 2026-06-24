"""Extracted math functions for numpy eager."""

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry

# pragma: no cover


@numpy_eager_registry.register("Sin")
def _np_sin(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.sin(*args, **kwargs)


@numpy_eager_registry.register("Cos")
def _np_cos(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.cos(*args, **kwargs)
