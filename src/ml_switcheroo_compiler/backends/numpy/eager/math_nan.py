# ruff: noqa: E501
"""Numpy eager NaN-safe operations."""

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("DivideNoNan")
def _np_divide_no_nan(backend_module: object, x: object, y: object, **kwargs: object) -> object:
    """DivideNoNan.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        y (object): The y parameter.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return backend_module.divide(x, y, out=backend_module.zeros_like(x, dtype=backend_module.result_type(x, y)), where=y != 0)


@numpy_eager_registry.register("MultiplyNoNan")
def _np_multiply_no_nan(backend_module: object, x: object, y: object, **kwargs: object) -> object:
    """MultiplyNoNan.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        y (object): The y parameter.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return backend_module.multiply(x, y, out=backend_module.zeros_like(x, dtype=backend_module.result_type(x, y)), where=y != 0)


@numpy_eager_registry.register("SquaredDifference")
def _np_squared_difference(backend_module: object, x: object, y: object, **kwargs: object) -> object:
    """SquaredDifference.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        y (object): The y parameter.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    diff = backend_module.subtract(x, y)
    return backend_module.square(diff)


@numpy_eager_registry.register("Xdivy")
def _np_xdivy(backend_module: object, x: object, y: object, **kwargs: object) -> object:
    """Xdivy.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        y (object): The y parameter.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return backend_module.divide(x, y, out=backend_module.zeros_like(x, dtype=backend_module.result_type(x, y)), where=x != 0)


@numpy_eager_registry.register("Xlog1py")
def _np_xlog1py(backend_module: object, x: object, y: object, **kwargs: object) -> object:
    """Xlog1py.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        y (object): The y parameter.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return backend_module.multiply(
        x,
        backend_module.log1p(y),
        out=backend_module.zeros_like(x, dtype=backend_module.result_type(x, y)),
        where=x != 0,
    )


@numpy_eager_registry.register("ReciprocalNoNan")
def _np_reciprocal_no_nan(backend_module: object, x: object, **kwargs: object) -> object:
    """ReciprocalNoNan.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return backend_module.divide(1.0, x, out=backend_module.zeros_like(x, dtype=backend_module.result_type(x, 1.0)), where=x != 0)


@numpy_eager_registry.register("ZeroFraction")
def _np_zero_fraction(backend_module: object, x: object, **kwargs: object) -> object:
    """ZeroFraction.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    num_zeros = backend_module.sum(backend_module.equal(x, 0))
    total_elements = backend_module.size(x)
    if total_elements == 0:
        return backend_module.array(float("nan"))
    return backend_module.array(backend_module.divide(num_zeros, total_elements).astype(float))


def _xlogy(x: object, y: object) -> object:
    """Evaluate _xlogy operation.

    Args:
        x (object): The x parameter.
        y (object): The y parameter.

    Returns:
        object: Result.
    """
    res = np.where(x == 0.0, 0.0, x * np.log(y))
    return res


@numpy_eager_registry.register("Nanmean")
def _np_nanmean(backend_module: object, a: object, axis: object = None, keepdims: object = False, *args: object, **kwargs: object) -> object:
    """Nanmean.

    Args:
        backend_module (object): The backend_module parameter.
        a (object): The a parameter.
        axis (object): The axis parameter.
        keepdims (object): The keepdims parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return backend_module.nanmean(a, axis=axis, keepdims=keepdims)


@numpy_eager_registry.register("Nanmedian")
def _np_nanmedian(backend_module: object, a: object, axis: object = None, keepdims: object = False, *args: object, **kwargs: object) -> object:
    """Nanmedian.

    Args:
        backend_module (object): The backend_module parameter.
        a (object): The a parameter.
        axis (object): The axis parameter.
        keepdims (object): The keepdims parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return backend_module.nanmedian(a, axis=axis, keepdims=keepdims)
