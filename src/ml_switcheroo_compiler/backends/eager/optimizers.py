# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Optimizer update operations for eager backends."""

from typing import Any

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry


@global_eager_registry.register("ApplyAdam")
def apply_adam(backend_module: Any, param: Any, m: Any, v: Any, grad: Any, lr: float) -> tuple[Any, Any, Any]:
    """Apply Adam update using backend operations.

    Args:
        backend_module (object): The backend_module parameter.
        param (object): The param parameter.
        m (object): The m parameter.
        v (object): The v parameter.
        grad (object): The grad parameter.
        lr (float): The lr parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m_new = backend_module.add(backend_module.multiply(m, beta1), backend_module.multiply(grad, 1.0 - beta1))
    v_new = backend_module.add(backend_module.multiply(v, beta2), backend_module.multiply(backend_module.multiply(grad, grad), 1.0 - beta2))
    update = backend_module.divide(m_new, backend_module.add(backend_module.sqrt(v_new), eps))
    p_new = backend_module.subtract(param, backend_module.multiply(update, lr))
    return p_new, m_new, v_new


@global_eager_registry.register("ApplyAdagrad")
def apply_adagrad(backend_module: Any, param: Any, accum: Any, grad: Any, lr: float) -> tuple[Any, Any]:
    """Apply Adagrad update using backend operations.

    Args:
        backend_module (object): The backend_module parameter.
        param (object): The param parameter.
        accum (object): The accum parameter.
        grad (object): The grad parameter.
        lr (float): The lr parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    accum_new = backend_module.add(accum, backend_module.multiply(grad, grad))
    update = backend_module.divide(grad, backend_module.add(backend_module.sqrt(accum_new), 1e-10))
    p_new = backend_module.subtract(param, backend_module.multiply(update, lr))
    return p_new, accum_new


@global_eager_registry.register("ApplyFtrl")
def apply_ftrl(backend_module: Any, param: Any, accum: Any, linear: Any, grad: Any, lr: float) -> tuple[Any, Any, Any]:
    """Apply FTRL update using backend operations.

    Args:
        backend_module (object): The backend_module parameter.
        param (object): The param parameter.
        accum (object): The accum parameter.
        linear (object): The linear parameter.
        grad (object): The grad parameter.
        lr (float): The lr parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    accum_new = backend_module.add(accum, backend_module.multiply(grad, grad))
    sigma = backend_module.divide(backend_module.subtract(backend_module.sqrt(accum_new), backend_module.sqrt(accum)), lr)
    linear_new = backend_module.add(backend_module.subtract(linear, grad), backend_module.multiply(sigma, param))
    p_new = backend_module.subtract(param, backend_module.multiply(grad, lr))
    return p_new, accum_new, linear_new


@global_eager_registry.register("ApplyRMSProp")
def apply_rmsprop(backend_module: Any, param: Any, ms: Any, mom: Any, grad: Any, lr: float) -> tuple[Any, Any, Any]:
    """Apply RMSProp update using backend operations.

    Args:
        backend_module (object): The backend_module parameter.
        param (object): The param parameter.
        ms (object): The ms parameter.
        mom (object): The mom parameter.
        grad (object): The grad parameter.
        lr (float): The lr parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    rho, momentum, eps = 0.9, 0.0, 1e-8
    ms_new = backend_module.add(backend_module.multiply(ms, rho), backend_module.multiply(backend_module.multiply(grad, grad), 1.0 - rho))
    mom_new = backend_module.add(backend_module.multiply(mom, momentum), backend_module.divide(grad, backend_module.add(backend_module.sqrt(ms_new), eps)))
    p_new = backend_module.subtract(param, backend_module.multiply(mom_new, lr))
    return p_new, ms_new, mom_new
