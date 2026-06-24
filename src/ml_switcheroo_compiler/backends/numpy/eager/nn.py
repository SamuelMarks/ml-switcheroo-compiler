"""Module docstring."""

from ml_switcheroo_compiler.core.constants import MAGIC_VAL_3

import math

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


def _gelu(x: object, *args: object, **kwargs: object) -> object:
    r"""Execute _gelu.\n\n    Args:\n        cls (Any): The class.\n        x (Any): Argument x.\n        *args (Any): Argument *args.\n        **kwargs (Any): Argument **kwargs.\n\n    Returns:\n    Any: The result.\n."""
    erf_vec = np.vectorize(math.erf)  # pragma: no cover
    return (0.5 * x) * (1 + erf_vec(x / np.sqrt(2.0)))  # pragma: no cover


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
    alpha = 1.6732632423543772848170429916717  # pragma: no cover
    scale = 1.0507009873554804934193349852946  # pragma: no cover
    alpha_p = -alpha * scale  # pragma: no cover

    import numpy as np  # pragma: no cover

    rng = np.random.default_rng(kwargs.get("seed", None))  # pragma: no cover
    noise_shape = kwargs.get("noise_shape", None)  # pragma: no cover
    if noise_shape is None:  # pragma: no cover
        noise_shape = x.shape  # pragma: no cover

    mask = rng.binomial(1, 1.0 - rate, size=noise_shape)  # pragma: no cover

    a = 1.0 / np.sqrt(1.0 - rate + rate * rate * alpha_p * alpha_p)  # pragma: no cover
    b = -a * alpha_p * rate  # pragma: no cover

    return a * (x * mask + alpha_p * (1.0 - mask)) + b  # pragma: no cover


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
    rate = kwargs.get("rate", 0.5)  # pragma: no cover
    training = kwargs.get("training", False)  # pragma: no cover
    if not training or rate == 0.0:  # pragma: no cover
        return x  # pragma: no cover

    import numpy as np  # pragma: no cover

    rng = np.random.default_rng(kwargs.get("seed", None))  # pragma: no cover
    noise_shape = kwargs.get("noise_shape", None)  # pragma: no cover
    if noise_shape is None:  # pragma: no cover
        noise_shape = x.shape  # pragma: no cover

    mask = rng.binomial(1, 1.0 - rate, size=noise_shape)  # pragma: no cover
    return (x * mask) / (1.0 - rate)  # pragma: no cover


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
    import numpy as np  # pragma: no cover

    shape = x.shape  # pragma: no cover
    if len(shape) < MAGIC_VAL_3:  # pragma: no cover
        return execute_generic_op(backend_module, wrapped_op_name, x, **kwargs)  # pragma: no cover

    flat_x = np.reshape(x, (shape[0] * shape[1], *shape[2:]))  # pragma: no cover
    out = execute_generic_op(backend_module, wrapped_op_name, flat_x, **kwargs)  # pragma: no cover
    out_shape = (shape[0], shape[1], *out.shape[1:])  # pragma: no cover
    return np.reshape(out, out_shape)  # pragma: no cover
