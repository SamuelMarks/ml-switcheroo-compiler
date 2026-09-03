# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Core abstractions and logic definitions for nn.py."""

import math

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.constants import MAGIC_VAL_3


def _gelu(x, *args, **kwargs):
    """Evaluate _gelu operation.

    Args:
        x (object): The x parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    erf_vec = np.vectorize(math.erf)
    return 0.5 * x * (1 + erf_vec(x / np.sqrt(2.0)))


@numpy_eager_registry.register("Relu")
def _np_relu(backend_module, x, *args, **kwargs):
    """Evaluate _np_relu operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.maximum(x, 0.0)


@numpy_eager_registry.register("AlphaDropout")
def _np_alpha_dropout(backend_module, x, **kwargs):
    """Evaluate _np_alpha_dropout operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    rate = kwargs.get("rate", 0.5)
    training = kwargs.get("training", False)
    if not training or rate == 0.0:
        return x
    alpha = 1.6732632423543772
    scale = 1.0507009873554805
    alpha_p = -alpha * scale
    rng = np.random.default_rng(kwargs.get("seed", None))
    noise_shape = kwargs.get("noise_shape", None)
    if noise_shape is None:
        noise_shape = x.shape
    mask = rng.binomial(1, 1.0 - rate, size=noise_shape)
    a = 1.0 / np.sqrt(1.0 - rate + rate * rate * alpha_p * alpha_p)
    b = -a * alpha_p * rate
    return a * (x * mask + alpha_p * (1.0 - mask)) + b


@numpy_eager_registry.register("ActivityRegularization")
def _np_activity_regularization(backend_module, x, **kwargs):
    """Evaluate _np_activity_regularization operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return x


@numpy_eager_registry.register("Dropout")
def _np_dropout(backend_module, x, rate=0.5, **kwargs):
    """Evaluate _np_dropout operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        rate (float): The dropout rate.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    rate = kwargs.get("rate", rate)
    training = kwargs.get("training", False)
    if not training or rate == 0.0:
        return x
    rng = np.random.default_rng(kwargs.get("seed", None))
    noise_shape = kwargs.get("noise_shape", None)
    if noise_shape is None:
        noise_shape = x.shape
    mask = rng.binomial(1, 1.0 - rate, size=noise_shape)
    return x * mask / (1.0 - rate)


@numpy_eager_registry.register("TimeDistributed")
def _np_time_distributed(backend_module, x, **kwargs):
    """Evaluate _np_time_distributed operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    wrapped_op_name = kwargs.pop("wrapped_op_name")
    shape = x.shape
    if len(shape) < MAGIC_VAL_3:
        return get_active_backend().execute_op(wrapped_op_name, x, **kwargs)
    flat_x = np.reshape(x, (shape[0] * shape[1], *shape[2:]))
    out = get_active_backend().execute_op(wrapped_op_name, flat_x, **kwargs)
    out_shape = (shape[0], shape[1], *out.shape[1:])
    return np.reshape(out, out_shape)


@numpy_eager_registry.register("Rope")
def _np_rope(backend_module, x, **kwargs):
    """Apply Rotary Positional Encoding using NumPy.

    Args:
        backend_module (object): Backend module.
        x (object): Input tensor.
        **kwargs (object): Keyword arguments.

    Returns: np.ndarray: Output tensor.
    """
    x_np = backend_module.asarray(x)
    half_dim = kwargs.get("axis", kwargs.get("dim", x_np.shape[-1])) // 2
    position = backend_module.arange(kwargs.get("offset", 0), kwargs.get("offset", 0) + x_np.shape[-2], dtype=x_np.dtype)
    freqs = backend_module.exp(-backend_module.arange(0, half_dim, dtype=x_np.dtype) * (backend_module.log(kwargs.get("base", 10000.0)) / half_dim))
    angles = position[:, None] * freqs[None, :]

    return backend_module.concatenate(
        [
            x_np[..., :half_dim] * backend_module.cos(angles) - x_np[..., half_dim:] * backend_module.sin(angles),
            x_np[..., :half_dim] * backend_module.sin(angles) + x_np[..., half_dim:] * backend_module.cos(angles),
        ],
        axis=-1,
    )


@numpy_eager_registry.register("Rrelu")
def _np_rrelu(backend_module, x, *args, **kwargs):
    """Evaluate _np_rrelu operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    lower = kwargs.get("lower", 1.0 / 8.0)
    upper = kwargs.get("upper", 1.0 / 3.0)
    training = kwargs.get("training", False)

    x_data = backend_module.asarray(getattr(x, "data", x))
    if not training:
        alpha = (lower + upper) / 2.0
        return backend_module.where(x_data >= 0, x_data, x_data * alpha)

    alpha = backend_module.random.uniform(lower, upper, size=x_data.shape)
    return backend_module.where(x_data >= 0, x_data, x_data * alpha)
