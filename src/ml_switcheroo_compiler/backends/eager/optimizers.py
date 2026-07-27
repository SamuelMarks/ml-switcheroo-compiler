"""Optimizer update operations for eager backends."""

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry


@global_eager_registry.register("ApplyAdam")
def apply_adam(backend_module: object, param: object, m: object, v: object, grad: object, lr: float) -> tuple[object, object, object]:
    """Apply Adam update using backend operations."""
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m_new = backend_module.add(backend_module.multiply(m, beta1), backend_module.multiply(grad, 1.0 - beta1))
    v_new = backend_module.add(backend_module.multiply(v, beta2), backend_module.multiply(backend_module.multiply(grad, grad), 1.0 - beta2))
    update = backend_module.divide(m_new, backend_module.add(backend_module.sqrt(v_new), eps))
    p_new = backend_module.subtract(param, backend_module.multiply(update, lr))
    return p_new, m_new, v_new


@global_eager_registry.register("ApplyAdagrad")
def apply_adagrad(backend_module: object, param: object, accum: object, grad: object, lr: float) -> tuple[object, object]:
    """Apply Adagrad update using backend operations."""
    accum_new = backend_module.add(accum, backend_module.multiply(grad, grad))
    update = backend_module.divide(grad, backend_module.add(backend_module.sqrt(accum_new), 1e-10))
    p_new = backend_module.subtract(param, backend_module.multiply(update, lr))
    return p_new, accum_new


@global_eager_registry.register("ApplyFtrl")
def apply_ftrl(backend_module: object, param: object, accum: object, linear: object, grad: object, lr: float) -> tuple[object, object, object]:
    """Apply FTRL update using backend operations."""
    accum_new = backend_module.add(accum, backend_module.multiply(grad, grad))
    sigma = backend_module.divide(backend_module.subtract(backend_module.sqrt(accum_new), backend_module.sqrt(accum)), lr)
    linear_new = backend_module.add(backend_module.subtract(linear, grad), backend_module.multiply(sigma, param))
    p_new = backend_module.subtract(param, backend_module.multiply(grad, lr))
    return p_new, accum_new, linear_new


@global_eager_registry.register("ApplyRMSProp")
def apply_rmsprop(backend_module: object, param: object, ms: object, mom: object, grad: object, lr: float) -> tuple[object, object, object]:
    """Apply RMSProp update using backend operations."""
    rho, momentum, eps = 0.9, 0.0, 1e-8
    ms_new = backend_module.add(backend_module.multiply(ms, rho), backend_module.multiply(backend_module.multiply(grad, grad), 1.0 - rho))
    mom_new = backend_module.add(backend_module.multiply(mom, momentum), backend_module.divide(grad, backend_module.add(backend_module.sqrt(ms_new), eps)))
    p_new = backend_module.subtract(param, backend_module.multiply(mom_new, lr))
    return p_new, ms_new, mom_new
