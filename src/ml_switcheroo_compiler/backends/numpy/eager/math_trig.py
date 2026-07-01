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


@numpy_eager_registry.register("Acos")
def _np_acos(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.arccos(*args, **kwargs)


@numpy_eager_registry.register("Acosh")
def _np_acosh(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.arccosh(*args, **kwargs)


@numpy_eager_registry.register("Asin")
def _np_asin(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.arcsin(*args, **kwargs)


@numpy_eager_registry.register("Asinh")
def _np_asinh(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.arcsinh(*args, **kwargs)


@numpy_eager_registry.register("Atan")
def _np_atan(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.arctan(*args, **kwargs)


@numpy_eager_registry.register("Atanh")
def _np_atanh(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.arctanh(*args, **kwargs)


@numpy_eager_registry.register("Atan2")
def _np_atan2(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.arctan2(*args, **kwargs)
