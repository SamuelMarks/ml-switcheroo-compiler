"""Eager optimizer step update implementations across backend modules."""

from __future__ import annotations

import types

from ml_switcheroo_compiler.backends.eager_registry import BackendArg, global_eager_registry
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module


@global_eager_registry.register("ApplyAdam")
def apply_adam(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    param: BackendArg,
    m: BackendArg,
    v: BackendArg,
    grad: BackendArg,
    lr: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    **kwargs: BackendArg,
) -> tuple[BackendArg, BackendArg, BackendArg]:
    """Apply an Adam optimizer parameter update step using backend array operations.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend array module.
        param (BackendArg): Parameter tensor to update.
        m (BackendArg): First moment running average tensor.
        v (BackendArg): Second moment running average tensor.
        grad (BackendArg): Gradient tensor with respect to parameter.
        lr (float): Learning rate scalar.
        beta1 (float): Exponential decay rate for first moment estimates.
        beta2 (float): Exponential decay rate for second moment estimates.
        **kwargs (BackendArg): Unused extra hyperparameter attributes.

    Returns:
        tuple[BackendArg, BackendArg, BackendArg]: Tuple of (updated_param, updated_m, updated_v).
    """
    del kwargs
    eps = 1e-8
    m_new = adapt_backend_module(backend_module).require_symbol("add")(
        adapt_backend_module(backend_module).require_symbol("multiply")(m, beta1),
        adapt_backend_module(backend_module).require_symbol("multiply")(grad, 1.0 - beta1),
    )
    v_new = adapt_backend_module(backend_module).require_symbol("add")(
        adapt_backend_module(backend_module).require_symbol("multiply")(v, beta2),
        adapt_backend_module(backend_module).require_symbol("multiply")(
            adapt_backend_module(backend_module).require_symbol("multiply")(grad, grad), 1.0 - beta2
        ),
    )
    update = adapt_backend_module(backend_module).require_symbol("divide")(
        m_new,
        adapt_backend_module(backend_module).require_symbol("add")(
            adapt_backend_module(backend_module).require_symbol("sqrt")(v_new), eps
        ),
    )
    p_new = adapt_backend_module(backend_module).require_symbol("subtract")(
        param, adapt_backend_module(backend_module).require_symbol("multiply")(update, lr)
    )
    return p_new, m_new, v_new


@global_eager_registry.register("ApplyAdagrad")
def apply_adagrad(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    param: BackendArg,
    accum: BackendArg,
    grad: BackendArg,
    lr: float = 0.01,
    eps: float = 1e-10,
    **kwargs: BackendArg,
) -> tuple[BackendArg, BackendArg]:
    """Apply an Adagrad optimizer parameter update step using backend array operations.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend array module.
        param (BackendArg): Parameter tensor to update.
        accum (BackendArg): Sum-of-squares gradient accumulator tensor.
        grad (BackendArg): Gradient tensor with respect to parameter.
        lr (float): Learning rate scalar.
        eps (float): Numerical stability constant.
        **kwargs (BackendArg): Unused extra hyperparameter attributes.

    Returns:
        tuple[BackendArg, BackendArg]: Tuple of (updated_param, updated_accum).
    """
    del kwargs
    accum_new = adapt_backend_module(backend_module).require_symbol("add")(
        accum, adapt_backend_module(backend_module).require_symbol("multiply")(grad, grad)
    )
    update = adapt_backend_module(backend_module).require_symbol("divide")(
        grad,
        adapt_backend_module(backend_module).require_symbol("add")(
            adapt_backend_module(backend_module).require_symbol("sqrt")(accum_new), eps
        ),
    )
    p_new = adapt_backend_module(backend_module).require_symbol("subtract")(
        param, adapt_backend_module(backend_module).require_symbol("multiply")(update, lr)
    )
    return p_new, accum_new


@global_eager_registry.register("ApplyFtrl")
def apply_ftrl(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    param: BackendArg,
    accum: BackendArg,
    linear: BackendArg,
    grad: BackendArg,
    lr: float = 0.001,
    **kwargs: BackendArg,
) -> tuple[BackendArg, BackendArg, BackendArg]:
    """Apply an FTRL-Proximal optimizer parameter update step using backend array operations.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend array module.
        param (BackendArg): Parameter tensor to update.
        accum (BackendArg): Quadratic gradient accumulator tensor.
        linear (BackendArg): Linear gradient accumulator tensor.
        grad (BackendArg): Gradient tensor with respect to parameter.
        lr (float): Learning rate scalar.
        **kwargs (BackendArg): Unused extra hyperparameter attributes.

    Returns:
        tuple[BackendArg, BackendArg, BackendArg]: Tuple of (updated_param, updated_accum, updated_linear).
    """
    del kwargs
    accum_new = adapt_backend_module(backend_module).require_symbol("add")(
        accum, adapt_backend_module(backend_module).require_symbol("multiply")(grad, grad)
    )
    sigma = adapt_backend_module(backend_module).require_symbol("divide")(
        adapt_backend_module(backend_module).require_symbol("subtract")(
            adapt_backend_module(backend_module).require_symbol("sqrt")(accum_new),
            adapt_backend_module(backend_module).require_symbol("sqrt")(accum),
        ),
        lr,
    )
    linear_new = adapt_backend_module(backend_module).require_symbol("subtract")(
        adapt_backend_module(backend_module).require_symbol("add")(linear, grad),
        adapt_backend_module(backend_module).require_symbol("multiply")(sigma, param),
    )
    denom = adapt_backend_module(backend_module).require_symbol("divide")(
        adapt_backend_module(backend_module).require_symbol("sqrt")(accum_new), lr
    )
    p_new = adapt_backend_module(backend_module).require_symbol("divide")(
        adapt_backend_module(backend_module).require_symbol("multiply")(linear_new, -1.0),
        adapt_backend_module(backend_module).require_symbol("add")(denom, 1e-10),
    )
    return p_new, accum_new, linear_new


@global_eager_registry.register("ApplyRMSProp")
def apply_rmsprop(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    param: BackendArg,
    ms: BackendArg,
    mom: BackendArg,
    grad: BackendArg,
    lr: float = 0.001,
    rho: float = 0.9,
    momentum: float = 0.0,
    **kwargs: BackendArg,
) -> tuple[BackendArg, BackendArg, BackendArg]:
    """Apply an RMSProp optimizer parameter update step using backend array operations.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend array module.
        param (BackendArg): Parameter tensor to update.
        ms (BackendArg): Mean-square running average tensor.
        mom (BackendArg): Momentum buffer tensor.
        grad (BackendArg): Gradient tensor with respect to parameter.
        lr (float): Learning rate scalar.
        rho (float): Discounting factor for incoming gradients.
        momentum (float): Momentum scalar factor.
        **kwargs (BackendArg): Unused extra hyperparameter attributes.

    Returns:
        tuple[BackendArg, BackendArg, BackendArg]: Tuple of (updated_param, updated_ms, updated_mom).
    """
    del kwargs
    eps = 1e-8
    ms_new = adapt_backend_module(backend_module).require_symbol("add")(
        adapt_backend_module(backend_module).require_symbol("multiply")(ms, rho),
        adapt_backend_module(backend_module).require_symbol("multiply")(
            adapt_backend_module(backend_module).require_symbol("multiply")(grad, grad), 1.0 - rho
        ),
    )
    mom_new = adapt_backend_module(backend_module).require_symbol("add")(
        adapt_backend_module(backend_module).require_symbol("multiply")(mom, momentum),
        adapt_backend_module(backend_module).require_symbol("divide")(
            grad,
            adapt_backend_module(backend_module).require_symbol("add")(
                adapt_backend_module(backend_module).require_symbol("sqrt")(ms_new), eps
            ),
        ),
    )
    p_new = adapt_backend_module(backend_module).require_symbol("subtract")(
        param, adapt_backend_module(backend_module).require_symbol("multiply")(mom_new, lr)
    )
    return p_new, ms_new, mom_new
