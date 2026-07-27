"""Numpy Activation Ops."""

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("Relu")
def _np_relu(backend_module: object, x: object, *args: object, **kwargs: object) -> object:
    """Evaluate the relu logic eagerly backed by NumPy.

    Args:
        backend_module: Required parameter for backend_module (e.g., numpy).
        x: The input array or scalar.
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        The evaluated output applying ReLU.
    """
    return backend_module.maximum(x, 0.0)


@numpy_eager_registry.register("Elu")
def _np_elu(backend_module: object, x: object, alpha: float = 1.0, **kwargs: object) -> object:
    """Evaluate the elu logic eagerly backed by NumPy.

    Args:
        backend_module: Required parameter for backend_module (e.g., numpy).
        x: The input array or scalar.
        alpha: The alpha parameter for ELU.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        The evaluated output applying ELU.
    """
    return backend_module.where(x > 0, x, alpha * (backend_module.exp(x) - 1.0))


@numpy_eager_registry.register("Celu")
def _np_celu(backend_module: object, x: object, alpha: float = 1.0, **kwargs: object) -> object:
    """Evaluate the celu logic eagerly backed by NumPy.

    Args:
        backend_module: Required parameter for backend_module (e.g., numpy).
        x: The input array or scalar.
        alpha: The alpha parameter for CELU.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        The evaluated output applying CELU.
    """
    return backend_module.maximum(0.0, x) + backend_module.minimum(0.0, alpha * (backend_module.exp(x / alpha) - 1.0))


@numpy_eager_registry.register("Softplus")
def _np_softplus(backend_module: object, x: object, **kwargs: object) -> object:
    """Evaluate the softplus logic eagerly backed by NumPy.

    Args:
        backend_module: Required parameter for backend_module (e.g., numpy).
        x: The input array or scalar.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        The evaluated output applying Softplus.
    """
    return backend_module.log1p(backend_module.exp(-backend_module.abs(x))) + backend_module.maximum(x, 0.0)


@numpy_eager_registry.register("Softsign")
def _np_softsign(backend_module: object, x: object, **kwargs: object) -> object:
    """Evaluate the softsign logic eagerly backed by NumPy.

    Args:
        backend_module: Required parameter for backend_module (e.g., numpy).
        x: The input array or scalar.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        The evaluated output applying Softsign.
    """
    return x / (1.0 + backend_module.abs(x))


@numpy_eager_registry.register("Mish")
def _np_mish(backend_module: object, x: object, **kwargs: object) -> object:
    """Evaluate the mish logic eagerly backed by NumPy.

    Args:
        backend_module: Required parameter for backend_module (e.g., numpy).
        x: The input array or scalar.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        The evaluated output applying Mish.
    """
    softplus_x = backend_module.log1p(backend_module.exp(-backend_module.abs(x))) + backend_module.maximum(x, 0.0)
    return x * backend_module.tanh(softplus_x)


@numpy_eager_registry.register("LogSigmoid")
def _np_log_sigmoid(backend_module: object, x: object, **kwargs: object) -> object:
    """Evaluate the log_sigmoid logic eagerly backed by NumPy.

    Args:
        backend_module: Required parameter for backend_module (e.g., numpy).
        x: The input array or scalar.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        The evaluated output applying LogSigmoid.
    """
    return -backend_module.log1p(backend_module.exp(-x))
