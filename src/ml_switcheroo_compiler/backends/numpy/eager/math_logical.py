# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Extracted logical functions for numpy eager."""

from typing import Any

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("NotEqual")
def _np_not_equal(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_not_equal operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.not_equal(*args, **kwargs)


@numpy_eager_registry.register("Greater")
def _np_greater(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_greater operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.greater(*args, **kwargs)


@numpy_eager_registry.register("GreaterEqual")
def _np_greater_equal(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_greater_equal operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.greater_equal(*args, **kwargs)


@numpy_eager_registry.register("Less")
def _np_less(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_less operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.less(*args, **kwargs)


@numpy_eager_registry.register("LessEqual")
def _np_less_equal(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_less_equal operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.less_equal(*args, **kwargs)


@numpy_eager_registry.register("LogicalAnd")
def _np_logical_and(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_logical_and operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.logical_and(*args, **kwargs)


@numpy_eager_registry.register("LogicalOr")
def _np_logical_or(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_logical_or operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.logical_or(*args, **kwargs)


@numpy_eager_registry.register("LogicalNot")
def _np_logical_not(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_logical_not operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.logical_not(*args, **kwargs)


@numpy_eager_registry.register("LogicalXor")
def _np_logical_xor(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_logical_xor operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.logical_xor(*args, **kwargs)


@numpy_eager_registry.register("Where")
def _np_where(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_where operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.where(*args, **kwargs)


@numpy_eager_registry.register("Assert")
def _np_assert(backend_module: Any, condition: Any, **kwargs: Any) -> Any:
    """Evaluate _np_assert operation.

    Args:
        backend_module (object): The backend_module parameter.
        condition (object): The condition parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.

    Raises:
        AssertionError: An exception.
    """
    if not np.all(condition):
        data = kwargs.get("data", ["Assertion failed."])
        raise AssertionError(data)
    return backend_module.array(0)
