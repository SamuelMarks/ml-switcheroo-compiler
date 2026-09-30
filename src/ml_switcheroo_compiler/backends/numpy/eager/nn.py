"""Core neural network operations for numpy eager backend."""

from __future__ import annotations

import math
from types import ModuleType
from typing import Union

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.constants import MAGIC_VAL_3


def _gelu(
    x: np.ndarray,
    *args: Union[float, int, None],
    **kwargs: Union[float, int, None],
) -> np.ndarray:
    """Evaluate Gaussian Error Linear Unit (GELU) activation.

    Args:
        x (np.ndarray): Input tensor.
        *args (Union[float, int, None]): Additional positional arguments.
        **kwargs (Union[float, int, None]): Additional keyword arguments.

    Returns:
        np.ndarray: GELU activated tensor.
    """
    erf_vec = np.vectorize(math.erf)
    return 0.5 * x * (1.0 + erf_vec(x / np.sqrt(2.0)))


@numpy_eager_registry.register("Relu")
def _np_relu(
    backend_module: ModuleType,
    x: np.ndarray,
    *args: Union[float, int, None],
    **kwargs: Union[float, int, None],
) -> np.ndarray:
    """Evaluate Rectified Linear Unit (ReLU) activation.

    Args:
        backend_module (ModuleType): Active backend module.
        x (np.ndarray): Input tensor.
        *args (Union[float, int, None]): Additional positional arguments.
        **kwargs (Union[float, int, None]): Additional keyword arguments.

    Returns:
        np.ndarray: ReLU activated tensor.
    """
    return backend_module.maximum(x, 0.0)


@numpy_eager_registry.register("AlphaDropout")
def _np_alpha_dropout(
    backend_module: ModuleType,
    x: np.ndarray,
    **kwargs: Union[float, bool, int, tuple[int, ...], None],
) -> np.ndarray:
    """Evaluate Alpha Dropout for self-normalizing neural networks.

    Args:
        backend_module (ModuleType): Active backend module.
        x (np.ndarray): Input tensor.
        **kwargs (Union[float, bool, int, tuple[int, ...], None]):
            Keyword arguments (rate, training, seed, noise_shape).

    Returns:
        np.ndarray: Output tensor with alpha dropout applied.
    """
    rate_val = kwargs.get("rate", 0.5)
    rate = float(rate_val) if rate_val is not None else 0.5
    training = bool(kwargs.get("training", False))
    if not training or rate == 0.0:
        return x
    alpha = 1.6732632423543772
    scale = 1.0507009873554805
    alpha_p = -alpha * scale
    seed_val = kwargs.get("seed", None)
    rng = np.random.default_rng(int(seed_val) if seed_val is not None else None)
    noise_shape_val = kwargs.get("noise_shape", None)
    noise_shape = noise_shape_val if isinstance(noise_shape_val, tuple) else x.shape
    mask = rng.binomial(1, 1.0 - rate, size=noise_shape)
    a = 1.0 / np.sqrt(1.0 - rate + rate * rate * alpha_p * alpha_p)
    b = -a * alpha_p * rate
    return a * (x * mask + alpha_p * (1.0 - mask)) + b


@numpy_eager_registry.register("ActivityRegularization")
def _np_activity_regularization(
    backend_module: ModuleType,
    x: np.ndarray,
    **kwargs: Union[float, None],
) -> np.ndarray:
    """Evaluate activity regularization passthrough.

    Args:
        backend_module (ModuleType): Active backend module.
        x (np.ndarray): Input tensor.
        **kwargs (Union[float, None]): Keyword arguments.

    Returns:
        np.ndarray: Unmodified input tensor.
    """
    return x


@numpy_eager_registry.register("Dropout")
def _np_dropout(
    backend_module: ModuleType,
    x: np.ndarray,
    rate: float = 0.5,
    **kwargs: Union[float, bool, int, tuple[int, ...], None],
) -> np.ndarray:
    """Evaluate inverted dropout on tensor.

    Args:
        backend_module (ModuleType): Active backend module.
        x (np.ndarray): Input tensor.
        rate (float): Dropout probability rate.
        **kwargs (Union[float, bool, int, tuple[int, ...], None]):
            Keyword arguments (rate, training, seed, noise_shape).

    Returns:
        np.ndarray: Dropout applied tensor.
    """
    rate_val = kwargs.get("rate", rate)
    rate_f = float(rate_val) if rate_val is not None else rate
    training = bool(kwargs.get("training", False))
    if not training or rate_f == 0.0:
        return x
    seed_val = kwargs.get("seed", None)
    rng = np.random.default_rng(int(seed_val) if seed_val is not None else None)
    noise_shape_val = kwargs.get("noise_shape", None)
    noise_shape = noise_shape_val if isinstance(noise_shape_val, tuple) else x.shape
    mask = rng.binomial(1, 1.0 - rate_f, size=noise_shape)
    return x * mask / (1.0 - rate_f)


@numpy_eager_registry.register("TimeDistributed")
def _np_time_distributed(
    backend_module: ModuleType,
    x: np.ndarray,
    **kwargs: Union[str, float, int, None],
) -> np.ndarray:
    """Apply an operation over time steps of a sequential tensor.

    Args:
        backend_module (ModuleType): Active backend module.
        x (np.ndarray): Input sequence tensor.
        **kwargs (Union[str, float, int, None]): Keyword arguments including wrapped_op_name.

    Returns:
        np.ndarray: Time distributed transformed tensor.
    """
    wrapped_op_name = str(kwargs.pop("wrapped_op_name"))
    shape = x.shape
    if len(shape) < MAGIC_VAL_3:
        return np.asarray(get_active_backend().execute_op(wrapped_op_name, x, **kwargs))
    flat_x = np.reshape(x, (shape[0] * shape[1], *shape[2:]))
    out = get_active_backend().execute_op(wrapped_op_name, flat_x, **kwargs)
    out_shape = (shape[0], shape[1], *out.shape[1:])
    return np.reshape(out, out_shape)


@numpy_eager_registry.register("Rope")
def _np_rope(
    backend_module: ModuleType,
    x: np.ndarray,
    **kwargs: Union[int, float, None],
) -> np.ndarray:
    """Apply Rotary Positional Encoding (RoPE) to embeddings.

    Args:
        backend_module (ModuleType): Active backend module.
        x (np.ndarray): Input tensor.
        **kwargs (Union[int, float, None]): Keyword arguments (dim, axis, offset, base).

    Returns:
        np.ndarray: RoPE encoded embeddings.
    """
    x_np = backend_module.asarray(x)
    dim_val = kwargs.get("axis", kwargs.get("dim", x_np.shape[-1]))
    dim_int = int(dim_val) if dim_val is not None else x_np.shape[-1]
    half_dim = dim_int // 2
    offset_val = kwargs.get("offset", 0)
    offset = int(offset_val) if offset_val is not None else 0
    base_val = kwargs.get("base", 10000.0)
    base = float(base_val) if base_val is not None else 10000.0

    position = backend_module.arange(offset, offset + x_np.shape[-2], dtype=x_np.dtype)
    freqs = backend_module.exp(-backend_module.arange(0, half_dim, dtype=x_np.dtype) * (backend_module.log(base) / half_dim))
    angles = position[:, None] * freqs[None, :]

    return backend_module.concatenate(
        [
            x_np[..., :half_dim] * backend_module.cos(angles) - x_np[..., half_dim:] * backend_module.sin(angles),
            x_np[..., :half_dim] * backend_module.sin(angles) + x_np[..., half_dim:] * backend_module.cos(angles),
        ],
        axis=-1,
    )


@numpy_eager_registry.register("Rrelu")
def _np_rrelu(
    backend_module: ModuleType,
    x: np.ndarray,
    *args: Union[float, int, None],
    **kwargs: Union[float, bool, None],
) -> np.ndarray:
    """Evaluate Randomized Leaky Rectified Linear Unit (RReLU).

    Args:
        backend_module (ModuleType): Active backend module.
        x (np.ndarray): Input tensor.
        *args (Union[float, int, None]): Additional positional arguments.
        **kwargs (Union[float, bool, None]): Keyword arguments (lower, upper, training).

    Returns:
        np.ndarray: RReLU activated tensor.
    """
    lower_val = kwargs.get("lower", 1.0 / 8.0)
    lower = float(lower_val) if lower_val is not None else 1.0 / 8.0
    upper_val = kwargs.get("upper", 1.0 / 3.0)
    upper = float(upper_val) if upper_val is not None else 1.0 / 3.0
    training = bool(kwargs.get("training", False))

    x_data = backend_module.asarray(getattr(x, "data", x))
    if not training:
        alpha = (lower + upper) / 2.0
        return backend_module.where(x_data >= 0, x_data, x_data * alpha)

    alpha = backend_module.random.uniform(lower, upper, size=x_data.shape)
    return backend_module.where(x_data >= 0, x_data, x_data * alpha)
