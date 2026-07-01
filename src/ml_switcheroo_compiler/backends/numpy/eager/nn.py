"""Module docstring."""

from ml_switcheroo_compiler.core.constants import MAGIC_VAL_3

import math

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


def _gelu(x: object, *args: object, **kwargs: object) -> object:
    r"""Execute _gelu.\n\n    Args:\n        cls (Any): The class.\n        x (Any): Argument x.\n        *args (Any): Argument *args.\n        **kwargs (Any): Argument **kwargs.\n\n    Returns:\n    Any: The result.\n."""
    erf_vec = np.vectorize(math.erf)
    return (0.5 * x) * (1 + erf_vec(x / np.sqrt(2.0)))


@numpy_eager_registry.register("Relu")
def _np_relu(backend_module: object, x: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        x: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.maximum(x, 0.0)


@numpy_eager_registry.register("AlphaDropout")
def _np_alpha_dropout(backend_module: object, x: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        x: Arg.
        kwargs: Arg.
    """
    rate = kwargs.get("rate", 0.5)
    training = kwargs.get("training", False)
    if not training or rate == 0.0:  # pragma: no branch
        return x

    # SELU parameters
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    alpha_p = -alpha * scale

    import numpy as np

    rng = np.random.default_rng(kwargs.get("seed", None))
    noise_shape = kwargs.get("noise_shape", None)
    if noise_shape is None:
        noise_shape = x.shape

    mask = rng.binomial(1, 1.0 - rate, size=noise_shape)

    a = 1.0 / np.sqrt(1.0 - rate + rate * rate * alpha_p * alpha_p)
    b = -a * alpha_p * rate

    return a * (x * mask + alpha_p * (1.0 - mask)) + b


@numpy_eager_registry.register("ActivityRegularization")
def _np_activity_regularization(backend_module: object, x: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        x: Arg.
        kwargs: Arg.
    """
    # Just returns x, as the regularization is injected into the loss
    return x


@numpy_eager_registry.register("Dropout")
def _np_dropout(backend_module: object, x: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        x: Arg.
        kwargs: Arg.
    """
    rate = kwargs.get("rate", 0.5)
    training = kwargs.get("training", False)
    if not training or rate == 0.0:
        return x

    import numpy as np

    rng = np.random.default_rng(kwargs.get("seed", None))
    noise_shape = kwargs.get("noise_shape", None)
    if noise_shape is None:
        noise_shape = x.shape

    mask = rng.binomial(1, 1.0 - rate, size=noise_shape)
    return (x * mask) / (1.0 - rate)


@numpy_eager_registry.register("TimeDistributed")
def _np_time_distributed(backend_module: object, x: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        x: Arg.
        kwargs: Arg.
    """
    from ml_switcheroo_compiler.backends.eager import execute_generic_op

    # Extract wrapped op
    wrapped_op_name = kwargs.pop("wrapped_op_name")

    # We flatten first two dims
    import numpy as np

    shape = x.shape
    if len(shape) < MAGIC_VAL_3:
        return execute_generic_op(backend_module, wrapped_op_name, x, **kwargs)

    flat_x = np.reshape(x, (shape[0] * shape[1], *shape[2:]))
    out = execute_generic_op(backend_module, wrapped_op_name, flat_x, **kwargs)
    out_shape = (shape[0], shape[1], *out.shape[1:])
    return np.reshape(out, out_shape)


@numpy_eager_registry.register("Rope")
def _np_rope(backend_module: object, x: object, **kwargs: object) -> object:
    """Apply Rotary Positional Encoding using NumPy."""
    dim = kwargs.get("dim")
    base = kwargs.get("base", 10000.0)
    offset = kwargs.get("offset", 0)

    x_np = backend_module.asarray(x)
    seq_len = x_np.shape[-2]

    # Generate positional indices
    position = backend_module.arange(offset, offset + seq_len, dtype=x_np.dtype)

    # Generate frequencies
    half_dim = dim // 2
    freqs = backend_module.exp(
        -backend_module.arange(0, half_dim, dtype=x_np.dtype)
        * (backend_module.log(base) / half_dim)
    )

    # Calculate angles
    angles = position[:, None] * freqs[None, :]

    # Apply RoPE
    sin_val = backend_module.sin(angles)
    cos_val = backend_module.cos(angles)

    x1 = x_np[..., :half_dim]
    x2 = x_np[..., half_dim:]

    rot_x1 = x1 * cos_val - x2 * sin_val
    rot_x2 = x1 * sin_val + x2 * cos_val

    return backend_module.concatenate([rot_x1, rot_x2], axis=-1)
