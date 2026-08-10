# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Numpy Activation Ops."""

from typing import Any

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("Relu")
def _np_relu(backend_module: Any, x: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_relu operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.maximum(x, 0.0)


@numpy_eager_registry.register("Elu")
def _np_elu(backend_module: Any, x: Any, alpha: float = 1.0, **kwargs: Any) -> Any:
    """Evaluate _np_elu operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        alpha (float): The alpha parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.where(x > 0, x, alpha * (backend_module.exp(x) - 1.0))


@numpy_eager_registry.register("Celu")
def _np_celu(backend_module: Any, x: Any, alpha: float = 1.0, **kwargs: Any) -> Any:
    """Evaluate _np_celu operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        alpha (float): The alpha parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.maximum(0.0, x) + backend_module.minimum(0.0, alpha * (backend_module.exp(x / alpha) - 1.0))


@numpy_eager_registry.register("Softplus")
def _np_softplus(backend_module: Any, x: Any, **kwargs: Any) -> Any:
    """Evaluate _np_softplus operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.log1p(backend_module.exp(-backend_module.abs(x))) + backend_module.maximum(x, 0.0)


@numpy_eager_registry.register("Softsign")
def _np_softsign(backend_module: Any, x: Any, **kwargs: Any) -> Any:
    """Evaluate _np_softsign operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return x / (1.0 + backend_module.abs(x))


@numpy_eager_registry.register("Mish")
def _np_mish(backend_module: Any, x: Any, **kwargs: Any) -> Any:
    """Evaluate _np_mish operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    softplus_x = backend_module.log1p(backend_module.exp(-backend_module.abs(x))) + backend_module.maximum(x, 0.0)
    return x * backend_module.tanh(softplus_x)


@numpy_eager_registry.register("LogSigmoid")
def _np_log_sigmoid(backend_module: Any, x: Any, **kwargs: Any) -> Any:
    """Evaluate _np_log_sigmoid operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return -backend_module.log1p(backend_module.exp(-x))
