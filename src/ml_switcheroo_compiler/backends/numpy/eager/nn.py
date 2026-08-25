# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Core abstractions and logic definitions for nn.py."""

import math

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.constants import MAGIC_VAL_3


def _gelu(x: object, *args: object, **kwargs: object) -> object:
    """Evaluate _gelu operation.

    Args:
        x (object): The x parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    erf_vec: object = np.vectorize(math.erf)
    return 0.5 * x * (1 + erf_vec(x / np.sqrt(2.0)))


@numpy_eager_registry.register("Relu")
def _np_relu(backend_module: object, x: object, *args: object, **kwargs: object) -> object:
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
def _np_alpha_dropout(backend_module: object, x: object, **kwargs: object) -> object:
    """Evaluate _np_alpha_dropout operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    rate: object = kwargs.get("rate", 0.5)
    training: object = kwargs.get("training", False)
    if not training or rate == 0.0:
        return x
    alpha: object = 1.6732632423543772
    scale: object = 1.0507009873554805
    alpha_p: object = -alpha * scale
    rng: object = np.random.default_rng(kwargs.get("seed", None))
    noise_shape: object = kwargs.get("noise_shape", None)
    if noise_shape is None:
        noise_shape: object = x.shape
    mask: object = rng.binomial(1, 1.0 - rate, size=noise_shape)
    a: object = 1.0 / np.sqrt(1.0 - rate + rate * rate * alpha_p * alpha_p)
    b: object = -a * alpha_p * rate
    return a * (x * mask + alpha_p * (1.0 - mask)) + b


@numpy_eager_registry.register("ActivityRegularization")
def _np_activity_regularization(backend_module: object, x: object, **kwargs: object) -> object:
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
def _np_dropout(backend_module: object, x: object, **kwargs: object) -> object:
    """Evaluate _np_dropout operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    rate: object = kwargs.get("rate", 0.5)
    training: object = kwargs.get("training", False)
    if not training or rate == 0.0:
        return x
    rng: object = np.random.default_rng(kwargs.get("seed", None))
    noise_shape: object = kwargs.get("noise_shape", None)
    if noise_shape is None:
        noise_shape: object = x.shape
    mask: object = rng.binomial(1, 1.0 - rate, size=noise_shape)
    return x * mask / (1.0 - rate)


@numpy_eager_registry.register("TimeDistributed")
def _np_time_distributed(backend_module: object, x: object, **kwargs: object) -> object:
    """Evaluate _np_time_distributed operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    wrapped_op_name: object = kwargs.pop("wrapped_op_name")
    shape: object = x.shape
    if len(shape) < MAGIC_VAL_3:
        return get_active_backend().execute_op(wrapped_op_name, x, **kwargs)
    flat_x: object = np.reshape(x, (shape[0] * shape[1], *shape[2:]))
    out: object = get_active_backend().execute_op(wrapped_op_name, flat_x, **kwargs)
    out_shape: object = (shape[0], shape[1], *out.shape[1:])
    return np.reshape(out, out_shape)


@numpy_eager_registry.register("Rope")
def _np_rope(backend_module: object, x: object, **kwargs: object) -> object:
    """Apply Rotary Positional Encoding using NumPy.

    Args:
        backend_module (object): Backend module.
        x (object): Input tensor.
        **kwargs (object): Keyword arguments.

    Returns: object: Output tensor.
    """
    x_np: object = backend_module.asarray(x)
    half_dim: object = kwargs.get("axis", kwargs.get("dim", x_np.shape[-1])) // 2
    position: object = backend_module.arange(kwargs.get("offset", 0), kwargs.get("offset", 0) + x_np.shape[-2], dtype=x_np.dtype)
    freqs: object = backend_module.exp(-backend_module.arange(0, half_dim, dtype=x_np.dtype) * (backend_module.log(kwargs.get("base", 10000.0)) / half_dim))
    angles: object = position[:, None] * freqs[None, :]

    return backend_module.concatenate(
        [
            x_np[..., :half_dim] * backend_module.cos(angles) - x_np[..., half_dim:] * backend_module.sin(angles),
            x_np[..., :half_dim] * backend_module.sin(angles) + x_np[..., half_dim:] * backend_module.cos(angles),
        ],
        axis=-1,
    )


@numpy_eager_registry.register("Rrelu")
def _np_rrelu(backend_module: object, x: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_rrelu operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    lower: object = kwargs.get("lower", 1.0 / 8.0)
    upper: object = kwargs.get("upper", 1.0 / 3.0)
    training: object = kwargs.get("training", False)

    x_data: object = backend_module.asarray(getattr(x, "data", x))
    if not training:
        alpha: object = (lower + upper) / 2.0
        return backend_module.where(x_data >= 0, x_data, x_data * alpha)

    alpha: object = backend_module.random.uniform(lower, upper, size=x_data.shape)
    return backend_module.where(x_data >= 0, x_data, x_data * alpha)
