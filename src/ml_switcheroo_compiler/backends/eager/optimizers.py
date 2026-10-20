"""Eager optimizer step update implementations across backend modules."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, global_eager_registry
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol


@global_eager_registry.register("ApplyAdam")
def apply_adam(
    backend_module: BackendModuleProtocol,
    param: EagerValue,
    m: EagerValue,
    v: EagerValue,
    grad: EagerValue,
    lr: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
) -> tuple[EagerValue, EagerValue, EagerValue]:
    """Apply an Adam optimizer parameter update step using backend array operations.

    Args:
        backend_module (ModuleType): Active backend array module.
        param (EagerValue): Parameter tensor to update.
        m (EagerValue): First moment running average tensor.
        v (EagerValue): Second moment running average tensor.
        grad (EagerValue): Gradient tensor with respect to parameter.
        lr (float): Learning rate scalar.
        beta1 (float): Exponential decay rate for first moment estimates.
        beta2 (float): Exponential decay rate for second moment estimates.

    Returns:
        tuple[EagerValue, EagerValue, EagerValue]: Tuple of (updated_param, updated_m, updated_v).
    """
    eps = 1e-8
    m_new = backend_module.add(backend_module.multiply(m, beta1), backend_module.multiply(grad, 1.0 - beta1))
    v_new = backend_module.add(backend_module.multiply(v, beta2), backend_module.multiply(backend_module.multiply(grad, grad), 1.0 - beta2))
    update = backend_module.divide(m_new, backend_module.add(backend_module.sqrt(v_new), eps))
    p_new = backend_module.subtract(param, backend_module.multiply(update, lr))
    return p_new, m_new, v_new


@global_eager_registry.register("ApplyAdagrad")
def apply_adagrad(
    backend_module: BackendModuleProtocol,
    param: EagerValue,
    accum: EagerValue,
    grad: EagerValue,
    lr: float = 0.01,
    eps: float = 1e-10,
) -> tuple[EagerValue, EagerValue]:
    """Apply an Adagrad optimizer parameter update step using backend array operations.

    Args:
        backend_module (ModuleType): Active backend array module.
        param (EagerValue): Parameter tensor to update.
        accum (EagerValue): Sum-of-squares gradient accumulator tensor.
        grad (EagerValue): Gradient tensor with respect to parameter.
        lr (float): Learning rate scalar.
        eps (float): Numerical stability constant.

    Returns:
        tuple[EagerValue, EagerValue]: Tuple of (updated_param, updated_accum).
    """
    accum_new = backend_module.add(accum, backend_module.multiply(grad, grad))
    update = backend_module.divide(grad, backend_module.add(backend_module.sqrt(accum_new), eps))
    p_new = backend_module.subtract(param, backend_module.multiply(update, lr))
    return p_new, accum_new


@global_eager_registry.register("ApplyFtrl")
def apply_ftrl(
    backend_module: BackendModuleProtocol,
    param: EagerValue,
    accum: EagerValue,
    linear: EagerValue,
    grad: EagerValue,
    lr: float = 0.001,
) -> tuple[EagerValue, EagerValue, EagerValue]:
    """Apply an FTRL-Proximal optimizer parameter update step using backend array operations.

    Args:
        backend_module (ModuleType): Active backend array module.
        param (EagerValue): Parameter tensor to update.
        accum (EagerValue): Quadratic gradient accumulator tensor.
        linear (EagerValue): Linear gradient accumulator tensor.
        grad (EagerValue): Gradient tensor with respect to parameter.
        lr (float): Learning rate scalar.

    Returns:
        tuple[EagerValue, EagerValue, EagerValue]: Tuple of (updated_param, updated_accum, updated_linear).
    """
    accum_new = backend_module.add(accum, backend_module.multiply(grad, grad))
    sigma = backend_module.divide(backend_module.subtract(backend_module.sqrt(accum_new), backend_module.sqrt(accum)), lr)
    linear_new = backend_module.subtract(backend_module.add(linear, grad), backend_module.multiply(sigma, param))
    denom = backend_module.divide(backend_module.sqrt(accum_new), lr)
    p_new = backend_module.divide(backend_module.multiply(linear_new, -1.0), backend_module.add(denom, 1e-10))
    return p_new, accum_new, linear_new


@global_eager_registry.register("ApplyRMSProp")
def apply_rmsprop(
    backend_module: BackendModuleProtocol,
    param: EagerValue,
    ms: EagerValue,
    mom: EagerValue,
    grad: EagerValue,
    lr: float = 0.001,
    rho: float = 0.9,
    momentum: float = 0.0,
) -> tuple[EagerValue, EagerValue, EagerValue]:
    """Apply an RMSProp optimizer parameter update step using backend array operations.

    Args:
        backend_module (ModuleType): Active backend array module.
        param (EagerValue): Parameter tensor to update.
        ms (EagerValue): Mean-square running average tensor.
        mom (EagerValue): Momentum buffer tensor.
        grad (EagerValue): Gradient tensor with respect to parameter.
        lr (float): Learning rate scalar.
        rho (float): Discounting factor for incoming gradients.
        momentum (float): Momentum scalar factor.

    Returns:
        tuple[EagerValue, EagerValue, EagerValue]: Tuple of (updated_param, updated_ms, updated_mom).
    """
    eps = 1e-8
    ms_new = backend_module.add(backend_module.multiply(ms, rho), backend_module.multiply(backend_module.multiply(grad, grad), 1.0 - rho))
    mom_new = backend_module.add(backend_module.multiply(mom, momentum), backend_module.divide(grad, backend_module.add(backend_module.sqrt(ms_new), eps)))
    p_new = backend_module.subtract(param, backend_module.multiply(mom_new, lr))
    return p_new, ms_new, mom_new
