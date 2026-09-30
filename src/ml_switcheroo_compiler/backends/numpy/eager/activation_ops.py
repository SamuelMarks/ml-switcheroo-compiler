"""Numpy Activation Ops."""

from __future__ import annotations

from types import ModuleType
from typing import Union

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("Relu")
def _np_relu(
    backend_module: ModuleType,
    x: np.ndarray,
    *args: Union[float, int, str, bool, None],
    **kwargs: Union[float, int, str, bool, None],
) -> np.ndarray:
    """Evaluate _np_relu operation.

    Args:
        backend_module (ModuleType): The backend module.
        x (np.ndarray): Input array.
        *args (Union[float, int, str, bool, None]): Positional args.
        **kwargs (Union[float, int, str, bool, None]): Keyword args.

    Returns:
        np.ndarray: Result.
    """
    return backend_module.maximum(x, 0.0)


@numpy_eager_registry.register("Elu")
def _np_elu(
    backend_module: ModuleType,
    x: np.ndarray,
    alpha: float = 1.0,
    **kwargs: Union[float, int, str, bool, None],
) -> np.ndarray:
    """Evaluate _np_elu operation.

    Args:
        backend_module (ModuleType): The backend module.
        x (np.ndarray): Input array.
        alpha (float): The alpha parameter.
        **kwargs (Union[float, int, str, bool, None]): Keyword args.

    Returns:
        np.ndarray: Result.
    """
    return backend_module.where(x > 0, x, alpha * (backend_module.exp(x) - 1.0))


@numpy_eager_registry.register("Celu")
def _np_celu(
    backend_module: ModuleType,
    x: np.ndarray,
    alpha: float = 1.0,
    **kwargs: Union[float, int, str, bool, None],
) -> np.ndarray:
    """Evaluate _np_celu operation.

    Args:
        backend_module (ModuleType): The backend module.
        x (np.ndarray): Input array.
        alpha (float): The alpha parameter.
        **kwargs (Union[float, int, str, bool, None]): Keyword args.

    Returns:
        np.ndarray: Result.
    """
    return backend_module.maximum(0.0, x) + backend_module.minimum(0.0, alpha * (backend_module.exp(x / alpha) - 1.0))


@numpy_eager_registry.register("Softplus")
def _np_softplus(
    backend_module: ModuleType,
    x: np.ndarray,
    **kwargs: Union[float, int, str, bool, None],
) -> np.ndarray:
    """Evaluate _np_softplus operation.

    Args:
        backend_module (ModuleType): The backend module.
        x (np.ndarray): Input array.
        **kwargs (Union[float, int, str, bool, None]): Keyword args.

    Returns:
        np.ndarray: Result.
    """
    return backend_module.log1p(backend_module.exp(-backend_module.abs(x))) + backend_module.maximum(x, 0.0)


@numpy_eager_registry.register("Softsign")
def _np_softsign(
    backend_module: ModuleType,
    x: np.ndarray,
    **kwargs: Union[float, int, str, bool, None],
) -> np.ndarray:
    """Evaluate _np_softsign operation.

    Args:
        backend_module (ModuleType): The backend module.
        x (np.ndarray): Input array.
        **kwargs (Union[float, int, str, bool, None]): Keyword args.

    Returns:
        np.ndarray: Result.
    """
    return x / (1.0 + backend_module.abs(x))


@numpy_eager_registry.register("Mish")
def _np_mish(
    backend_module: ModuleType,
    x: np.ndarray,
    **kwargs: Union[float, int, str, bool, None],
) -> np.ndarray:
    """Evaluate _np_mish operation.

    Args:
        backend_module (ModuleType): The backend module.
        x (np.ndarray): Input array.
        **kwargs (Union[float, int, str, bool, None]): Keyword args.

    Returns:
        np.ndarray: Result.
    """
    softplus_x = backend_module.log1p(backend_module.exp(-backend_module.abs(x))) + backend_module.maximum(x, 0.0)
    return x * backend_module.tanh(softplus_x)


@numpy_eager_registry.register("LogSigmoid")
def _np_log_sigmoid(
    backend_module: ModuleType,
    x: np.ndarray,
    **kwargs: Union[float, int, str, bool, None],
) -> np.ndarray:
    """Evaluate _np_log_sigmoid operation.

    Args:
        backend_module (ModuleType): The backend module.
        x (np.ndarray): Input array.
        **kwargs (Union[float, int, str, bool, None]): Keyword args.

    Returns:
        np.ndarray: Result.
    """
    return -backend_module.log1p(backend_module.exp(-x))
