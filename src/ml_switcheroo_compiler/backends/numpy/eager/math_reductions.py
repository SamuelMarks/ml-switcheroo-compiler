"""Extracted reduction functions for numpy eager."""

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry

# pragma: no cover


@numpy_eager_registry.register("Sum")
def _np_sum(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.sum(*args, **kwargs)


@numpy_eager_registry.register("Mean")
def _np_mean(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.mean(*args, **kwargs)


@numpy_eager_registry.register("Max")
def _np_max(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.max(*args, **kwargs)


@numpy_eager_registry.register("Min")
def _np_min(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.min(*args, **kwargs)


@numpy_eager_registry.register("Variance")
def _np_variance(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.var(*args, **kwargs)


@numpy_eager_registry.register("Std")
def _np_std(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.std(*args, **kwargs)


@numpy_eager_registry.register("Argmax")
def _np_argmax(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.argmax(*args, **kwargs)


@numpy_eager_registry.register("Argmin")
def _np_argmin(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.argmin(*args, **kwargs)


@numpy_eager_registry.register("Prod")
def _np_prod(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.prod(*args, **kwargs)


@numpy_eager_registry.register("All")
def _np_all(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.all(*args, **kwargs)


@numpy_eager_registry.register("AnyOp")
def _np_any_op(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.any(*args, **kwargs)  # pragma: no cover


@numpy_eager_registry.register("CountNonzero")
def _np_count_nonzero(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.count_nonzero(*args, **kwargs)


@numpy_eager_registry.register("Cumsum")
def _np_cumsum(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.cumsum(*args, **kwargs)


@numpy_eager_registry.register("AddN")
def _np_add_n(backend_module: object, inputs: list, **kwargs: object) -> object:
    """Execute _np_add_n.

    Args:
        backend_module: The backend module.
        inputs: List of tensors.
        kwargs: Additional arguments.

    Returns:
        The sum of the tensors.
    """
    if not inputs:
        raise ValueError("inputs must not be empty")  # pragma: no cover
    res = inputs[0]
    for i in range(1, len(inputs)):
        res = backend_module.add(res, inputs[i])
    return res


@numpy_eager_registry.register("AccumulateN")
def _np_accumulate_n(backend_module: object, inputs: list, **kwargs: object) -> object:
    """Execute _np_accumulate_n.

    Args:
        backend_module: The backend module.
        inputs: List of tensors.
        kwargs: Additional arguments.

    Returns:
        The accumulated sum.
    """
    if not inputs:
        raise ValueError("inputs must not be empty")  # pragma: no cover
    res = inputs[0]
    for i in range(1, len(inputs)):
        res = backend_module.add(res, inputs[i])
    return res


@numpy_eager_registry.register("CumulativeLogsumexp")
def _np_cumulative_logsumexp(
    backend_module: object, x: object, axis: int = 0, **kwargs: object
) -> object:
    """Execute _np_cumulative_logsumexp.

    Args:
        backend_module: The backend module.
        x: The input tensor.
        axis: The axis to compute.
        kwargs: Additional arguments.

    Returns:
        The cumulative logsumexp.
    """
    # log(cumsum(exp(x)))
    exp_x = backend_module.exp(x)
    cumsum_exp = backend_module.cumsum(exp_x, axis=axis)
    return backend_module.log(cumsum_exp)
