"""Extracted reduction functions for numpy eager backend."""

from __future__ import annotations

from collections.abc import Sequence
from types import ModuleType
from typing import Union

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("Sum")
def _np_sum(
    backend_module: ModuleType,
    *args: Union[np.ndarray, int, float, tuple[int, ...], None],
    **kwargs: Union[int, float, tuple[int, ...], bool, None],
) -> np.ndarray:
    """Evaluate sum reduction.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, int, float, tuple[int, ...], None]): Positional arguments to pass to sum.
        **kwargs (Union[int, float, tuple[int, ...], bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Computed sum reduction.
    """
    return backend_module.sum(*args, **kwargs)


@numpy_eager_registry.register("Mean")
def _np_mean(
    backend_module: ModuleType,
    *args: Union[np.ndarray, int, float, tuple[int, ...], None],
    **kwargs: Union[int, float, tuple[int, ...], bool, None],
) -> np.ndarray:
    """Evaluate mean reduction.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, int, float, tuple[int, ...], None]): Positional arguments to pass to mean.
        **kwargs (Union[int, float, tuple[int, ...], bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Computed mean reduction.
    """
    return backend_module.mean(*args, **kwargs)


@numpy_eager_registry.register("Max")
def _np_max(
    backend_module: ModuleType,
    *args: Union[np.ndarray, int, float, tuple[int, ...], None],
    **kwargs: Union[int, float, tuple[int, ...], bool, None],
) -> np.ndarray:
    """Evaluate max reduction.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, int, float, tuple[int, ...], None]): Positional arguments to pass to max.
        **kwargs (Union[int, float, tuple[int, ...], bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Computed max reduction.
    """
    return backend_module.max(*args, **kwargs)


@numpy_eager_registry.register("Min")
def _np_min(
    backend_module: ModuleType,
    *args: Union[np.ndarray, int, float, tuple[int, ...], None],
    **kwargs: Union[int, float, tuple[int, ...], bool, None],
) -> np.ndarray:
    """Evaluate min reduction.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, int, float, tuple[int, ...], None]): Positional arguments to pass to min.
        **kwargs (Union[int, float, tuple[int, ...], bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Computed min reduction.
    """
    return backend_module.min(*args, **kwargs)


@numpy_eager_registry.register("Variance")
def _np_variance(
    backend_module: ModuleType,
    *args: Union[np.ndarray, int, float, tuple[int, ...], None],
    **kwargs: Union[int, float, tuple[int, ...], bool, None],
) -> np.ndarray:
    """Evaluate variance reduction.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, int, float, tuple[int, ...], None]): Positional arguments to pass to var.
        **kwargs (Union[int, float, tuple[int, ...], bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Computed variance reduction.
    """
    kwargs.setdefault("ddof", 0)
    return backend_module.var(*args, **kwargs)


@numpy_eager_registry.register("Std")
def _np_std(
    backend_module: ModuleType,
    *args: Union[np.ndarray, int, float, tuple[int, ...], None],
    **kwargs: Union[int, float, tuple[int, ...], bool, None],
) -> np.ndarray:
    """Evaluate standard deviation reduction.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, int, float, tuple[int, ...], None]): Positional arguments to pass to std.
        **kwargs (Union[int, float, tuple[int, ...], bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Computed standard deviation reduction.
    """
    return backend_module.std(*args, **kwargs)


@numpy_eager_registry.register("Argmax")
def _np_argmax(
    backend_module: ModuleType,
    *args: Union[np.ndarray, int, float, tuple[int, ...], None],
    **kwargs: Union[int, float, tuple[int, ...], bool, None],
) -> np.ndarray:
    """Evaluate argmax reduction.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, int, float, tuple[int, ...], None]): Positional arguments to pass to argmax.
        **kwargs (Union[int, float, tuple[int, ...], bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Indices of maximum values.
    """
    return backend_module.argmax(*args, **kwargs)


@numpy_eager_registry.register("Argmin")
def _np_argmin(
    backend_module: ModuleType,
    *args: Union[np.ndarray, int, float, tuple[int, ...], None],
    **kwargs: Union[int, float, tuple[int, ...], bool, None],
) -> np.ndarray:
    """Evaluate argmin reduction.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, int, float, tuple[int, ...], None]): Positional arguments to pass to argmin.
        **kwargs (Union[int, float, tuple[int, ...], bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Indices of minimum values.
    """
    return backend_module.argmin(*args, **kwargs)


@numpy_eager_registry.register("Prod")
def _np_prod(
    backend_module: ModuleType,
    *args: Union[np.ndarray, int, float, tuple[int, ...], None],
    **kwargs: Union[int, float, tuple[int, ...], bool, None],
) -> np.ndarray:
    """Evaluate product reduction.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, int, float, tuple[int, ...], None]): Positional arguments to pass to prod.
        **kwargs (Union[int, float, tuple[int, ...], bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Computed product reduction.
    """
    return backend_module.prod(*args, **kwargs)


@numpy_eager_registry.register("AnyOp")
def _np_any_op(
    backend_module: ModuleType,
    *args: Union[np.ndarray, int, float, tuple[int, ...], None],
    **kwargs: Union[int, float, tuple[int, ...], bool, None],
) -> np.ndarray:
    """Evaluate any boolean reduction.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, int, float, tuple[int, ...], None]): Positional arguments to pass to any.
        **kwargs (Union[int, float, tuple[int, ...], bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Boolean reduction result.
    """
    return backend_module.any(*args, **kwargs)


@numpy_eager_registry.register("Cumsum")
def _np_cumsum(
    backend_module: ModuleType,
    *args: Union[np.ndarray, int, float, tuple[int, ...], None],
    **kwargs: Union[int, float, tuple[int, ...], bool, None],
) -> np.ndarray:
    """Evaluate cumulative sum.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, int, float, tuple[int, ...], None]): Positional arguments to pass to cumsum.
        **kwargs (Union[int, float, tuple[int, ...], bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Cumulative sum array.
    """
    return backend_module.cumsum(*args, **kwargs)


@numpy_eager_registry.register("AddN")
def _np_add_n(
    backend_module: ModuleType,
    inputs: Sequence[np.ndarray],
    **kwargs: Union[int, float, bool, None],
) -> np.ndarray:
    """Evaluate addition of a sequence of tensors.

    Args:
        backend_module (ModuleType): Active backend module.
        inputs (Sequence[np.ndarray]): List of tensors to sum.
        **kwargs (Union[int, float, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Element-wise sum of all tensors.

    Raises:
        ValueError: If inputs sequence is empty.
    """
    if not inputs:
        raise ValueError("inputs must not be empty")
    res = inputs[0]
    for i in range(1, len(inputs)):
        res = backend_module.add(res, inputs[i])
    return res


@numpy_eager_registry.register("AccumulateN")
def _np_accumulate_n(
    backend_module: ModuleType,
    inputs: Sequence[np.ndarray],
    **kwargs: Union[int, float, bool, None],
) -> np.ndarray:
    """Evaluate accumulation of a sequence of tensors.

    Args:
        backend_module (ModuleType): Active backend module.
        inputs (Sequence[np.ndarray]): List of tensors to accumulate.
        **kwargs (Union[int, float, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Element-wise sum of all tensors.

    Raises:
        ValueError: If inputs sequence is empty.
    """
    if not inputs:
        raise ValueError("inputs must not be empty")
    res = inputs[0]
    for i in range(1, len(inputs)):
        res = backend_module.add(res, inputs[i])
    return res


@numpy_eager_registry.register("CumulativeLogsumexp")
def _np_cumulative_logsumexp(
    backend_module: ModuleType,
    x: np.ndarray,
    axis: int = 0,
    **kwargs: Union[int, float, bool, None],
) -> np.ndarray:
    """Evaluate cumulative logsumexp along a specified axis.

    Args:
        backend_module (ModuleType): Active backend module.
        x (np.ndarray): Input tensor.
        axis (int): Axis along which to compute cumulative logsumexp.
        **kwargs (Union[int, float, bool, None]): Keyword arguments.

    Returns:
        np.ndarray: Result of cumulative log-sum-exp.
    """
    exp_x = backend_module.exp(x)
    cumsum_exp = backend_module.cumsum(exp_x, axis=axis)
    return backend_module.log(cumsum_exp)
