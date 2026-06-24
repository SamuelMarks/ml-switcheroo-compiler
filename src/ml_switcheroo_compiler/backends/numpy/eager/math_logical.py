"""Extracted logical functions for numpy eager."""

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry

# pragma: no cover


@numpy_eager_registry.register("NotEqual")
def _np_not_equal(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.not_equal(*args, **kwargs)


@numpy_eager_registry.register("Greater")
def _np_greater(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.greater(*args, **kwargs)


@numpy_eager_registry.register("GreaterEqual")
def _np_greater_equal(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.greater_equal(*args, **kwargs)


@numpy_eager_registry.register("Less")
def _np_less(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.less(*args, **kwargs)


@numpy_eager_registry.register("LessEqual")
def _np_less_equal(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.less_equal(*args, **kwargs)


@numpy_eager_registry.register("LogicalAnd")
def _np_logical_and(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.logical_and(*args, **kwargs)


@numpy_eager_registry.register("LogicalOr")
def _np_logical_or(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.logical_or(*args, **kwargs)


@numpy_eager_registry.register("LogicalNot")
def _np_logical_not(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.logical_not(*args, **kwargs)


@numpy_eager_registry.register("LogicalXor")
def _np_logical_xor(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.logical_xor(*args, **kwargs)


@numpy_eager_registry.register("Where")
def _np_where(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.where(*args, **kwargs)
