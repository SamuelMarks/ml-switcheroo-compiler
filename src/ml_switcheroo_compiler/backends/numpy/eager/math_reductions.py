# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Extracted reduction functions for numpy eager."""

from typing import Any

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("Sum")
def _np_sum(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_sum operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.sum(*args, **kwargs)


@numpy_eager_registry.register("Mean")
def _np_mean(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_mean operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.mean(*args, **kwargs)


@numpy_eager_registry.register("Max")
def _np_max(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_max operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.max(*args, **kwargs)


@numpy_eager_registry.register("Min")
def _np_min(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_min operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.min(*args, **kwargs)


@numpy_eager_registry.register("Variance")
def _np_variance(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_variance operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    kwargs.setdefault("ddof", 0)
    return backend_module.var(*args, **kwargs)


@numpy_eager_registry.register("Std")
def _np_std(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_std operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.std(*args, **kwargs)


@numpy_eager_registry.register("Argmax")
def _np_argmax(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_argmax operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.argmax(*args, **kwargs)


@numpy_eager_registry.register("Argmin")
def _np_argmin(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_argmin operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.argmin(*args, **kwargs)


@numpy_eager_registry.register("Prod")
def _np_prod(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_prod operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.prod(*args, **kwargs)


@numpy_eager_registry.register("AnyOp")
def _np_any_op(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_any_op operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.any(*args, **kwargs)


@numpy_eager_registry.register("Cumsum")
def _np_cumsum(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_cumsum operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.cumsum(*args, **kwargs)


@numpy_eager_registry.register("AddN")
def _np_add_n(backend_module: Any, inputs: list, **kwargs: Any) -> Any:
    """Evaluate _np_add_n operation.

    Args:
        backend_module (object): The backend_module parameter.
        inputs (list): The inputs parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.

    Raises:
        ValueError: An exception.
    """
    if not inputs:
        raise ValueError("inputs must not be empty")
    res = inputs[0]
    for i in range(1, len(inputs)):
        res = backend_module.add(res, inputs[i])
    return res


@numpy_eager_registry.register("AccumulateN")
def _np_accumulate_n(backend_module: Any, inputs: list, **kwargs: Any) -> Any:
    """Evaluate _np_accumulate_n operation.

    Args:
        backend_module (object): The backend_module parameter.
        inputs (list): The inputs parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.

    Raises:
        ValueError: An exception.
    """
    if not inputs:
        raise ValueError("inputs must not be empty")
    res = inputs[0]
    for i in range(1, len(inputs)):
        res = backend_module.add(res, inputs[i])
    return res


@numpy_eager_registry.register("CumulativeLogsumexp")
def _np_cumulative_logsumexp(backend_module: Any, x: Any, axis: int = 0, **kwargs: Any) -> Any:
    """Evaluate _np_cumulative_logsumexp operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        axis (int): The axis parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    exp_x = backend_module.exp(x)
    cumsum_exp = backend_module.cumsum(exp_x, axis=axis)
    return backend_module.log(cumsum_exp)
