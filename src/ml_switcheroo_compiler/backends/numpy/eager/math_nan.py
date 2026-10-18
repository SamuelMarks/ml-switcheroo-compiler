"""Numpy eager NaN-safe operations."""

from __future__ import annotations

from types import ModuleType
from typing import cast

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, numpy_eager_registry


@numpy_eager_registry.register("DivideNoNan")
def _np_divide_no_nan(backend_module: ModuleType, x: EagerValue, y: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """DivideNoNan.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        x (EagerValue): The x parameter.
        y (EagerValue): The y parameter.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, backend_module.divide(x, y, out=backend_module.zeros_like(x, dtype=backend_module.result_type(x, y)), where=y != 0))


@numpy_eager_registry.register("MultiplyNoNan")
def _np_multiply_no_nan(backend_module: ModuleType, x: EagerValue, y: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """MultiplyNoNan.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        x (EagerValue): The x parameter.
        y (EagerValue): The y parameter.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, backend_module.multiply(x, y, out=backend_module.zeros_like(x, dtype=backend_module.result_type(x, y)), where=y != 0))


@numpy_eager_registry.register("SquaredDifference")
def _np_squared_difference(backend_module: ModuleType, x: EagerValue, y: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """SquaredDifference.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        x (EagerValue): The x parameter.
        y (EagerValue): The y parameter.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    diff = backend_module.subtract(x, y)
    return cast(EagerValue, backend_module.square(diff))


@numpy_eager_registry.register("Xdivy")
def _np_xdivy(backend_module: ModuleType, x: EagerValue, y: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Xdivy.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        x (EagerValue): The x parameter.
        y (EagerValue): The y parameter.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, backend_module.divide(x, y, out=backend_module.zeros_like(x, dtype=backend_module.result_type(x, y)), where=x != 0))


@numpy_eager_registry.register("Xlog1py")
def _np_xlog1py(backend_module: ModuleType, x: EagerValue, y: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Xlog1py.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        x (EagerValue): The x parameter.
        y (EagerValue): The y parameter.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(
        EagerValue,
        backend_module.multiply(
            x,
            backend_module.log1p(y),
            out=backend_module.zeros_like(x, dtype=backend_module.result_type(x, y)),
            where=x != 0,
        ),
    )


@numpy_eager_registry.register("ReciprocalNoNan")
def _np_reciprocal_no_nan(backend_module: ModuleType, x: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """ReciprocalNoNan.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        x (EagerValue): The x parameter.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, backend_module.divide(1.0, x, out=backend_module.zeros_like(x, dtype=backend_module.result_type(x, 1.0)), where=x != 0))


@numpy_eager_registry.register("ZeroFraction")
def _np_zero_fraction(backend_module: ModuleType, x: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """ZeroFraction.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        x (EagerValue): The x parameter.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    num_zeros = backend_module.sum(backend_module.equal(x, 0))
    total_elements = backend_module.size(x)
    if total_elements == 0:
        return cast(EagerValue, backend_module.array(float("nan")))
    return cast(EagerValue, backend_module.array(backend_module.divide(num_zeros, total_elements).astype(float)))


def _xlogy(x: EagerValue, y: EagerValue) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Execute the eager xlogy operation on the active backend array.

    Args:
        x (EagerValue): The x parameter.
        y (EagerValue): The y parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    res = getattr(np, "where")(x == 0.0, 0.0, x * getattr(np, "log")(y))
    return cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], res)


@numpy_eager_registry.register("Nanmean")
def _np_nanmean(backend_module: ModuleType, a: EagerValue, axis: EagerValue = None, keepdims: EagerValue = False, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Nanmean.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        a (EagerValue): The a parameter.
        axis (EagerValue): The axis parameter.
        keepdims (EagerValue): The keepdims parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, backend_module.nanmean(a, axis=axis, keepdims=keepdims))


@numpy_eager_registry.register("Nanmedian")
def _np_nanmedian(backend_module: ModuleType, a: EagerValue, axis: EagerValue = None, keepdims: EagerValue = False, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Nanmedian.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        a (EagerValue): The a parameter.
        axis (EagerValue): The axis parameter.
        keepdims (EagerValue): The keepdims parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, backend_module.nanmedian(a, axis=axis, keepdims=keepdims))
