"""Numpy Activation Ops."""

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("Relu")
def _np_relu(backend_module: object, x: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_relu operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return backend_module.maximum(x, 0.0)


@numpy_eager_registry.register("Elu")
def _np_elu(backend_module: object, x: object, alpha: float = 1.0, **kwargs: object) -> object:
    """Evaluate _np_elu operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        alpha (float): The alpha parameter.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return backend_module.where(x > 0, x, alpha * (backend_module.exp(x) - 1.0))


@numpy_eager_registry.register("Celu")
def _np_celu(backend_module: object, x: object, alpha: float = 1.0, **kwargs: object) -> object:
    """Evaluate _np_celu operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        alpha (float): The alpha parameter.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return backend_module.maximum(0.0, x) + backend_module.minimum(0.0, alpha * (backend_module.exp(x / alpha) - 1.0))


@numpy_eager_registry.register("Softplus")
def _np_softplus(backend_module: object, x: object, **kwargs: object) -> object:
    """Evaluate _np_softplus operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return backend_module.log1p(backend_module.exp(-backend_module.abs(x))) + backend_module.maximum(x, 0.0)


@numpy_eager_registry.register("Softsign")
def _np_softsign(backend_module: object, x: object, **kwargs: object) -> object:
    """Evaluate _np_softsign operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return x / (1.0 + backend_module.abs(x))


@numpy_eager_registry.register("Mish")
def _np_mish(backend_module: object, x: object, **kwargs: object) -> object:
    """Evaluate _np_mish operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    softplus_x = backend_module.log1p(backend_module.exp(-backend_module.abs(x))) + backend_module.maximum(x, 0.0)
    return x * backend_module.tanh(softplus_x)


@numpy_eager_registry.register("LogSigmoid")
def _np_log_sigmoid(backend_module: object, x: object, **kwargs: object) -> object:
    """Evaluate _np_log_sigmoid operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return -backend_module.log1p(backend_module.exp(-x))
