# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Numpy implementations for optimizer update steps."""

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("ApplyAdam")
def _np_apply_adam(backend_module: object, param: object, m: object, v: object, grad: object, **kwargs: object) -> tuple[object, ...]:
    """Numpy eager fallback for ApplyAdam.

    Args:
        backend_module: The active backend module.
        param: The parameter array to update.
        m: The first moment array.
        v: The second moment array.
        grad: The gradient array.
        **kwargs: Additional hyperparameters like lr.

    Returns:
        A tuple of (updated_param, updated_m, updated_v).
    """
    p, m_, v_, g = np.asarray(param), np.asarray(m), np.asarray(v), np.asarray(grad)
    lr: object = kwargs.get("lr", 0.001)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m_new: object = m_ * beta1 + g * (1.0 - beta1)
    v_new: object = v_ * beta2 + (g**2) * (1.0 - beta2)
    update: object = m_new / (np.sqrt(v_new) + eps)
    p_new: object = p - lr * update
    return p_new, m_new, v_new


@numpy_eager_registry.register("ApplyAdagrad")
def _np_apply_adagrad(backend_module: object, param: object, accum: object, grad: object, **kwargs: object) -> tuple[object, ...]:
    """Numpy eager fallback for ApplyAdagrad.

    Args:
        backend_module: The active backend module.
        param: The parameter array to update.
        accum: The accumulation array.
        grad: The gradient array.
        **kwargs: Additional hyperparameters like lr.

    Returns:
        A tuple of (updated_param, updated_accum).
    """
    p, a, g = np.asarray(param), np.asarray(accum), np.asarray(grad)
    lr: object = kwargs.get("lr", 0.01)
    a_new: object = a + g**2
    update: object = g / (np.sqrt(a_new) + 1e-10)
    p_new: object = p - lr * update
    return p_new, a_new


@numpy_eager_registry.register("ApplyFtrl")
def _np_apply_ftrl(backend_module: object, param: object, accum: object, linear: object, grad: object, **kwargs: object) -> tuple[object, ...]:
    """Numpy eager fallback for ApplyFtrl.

    Args:
        backend_module: The active backend module.
        param: The parameter array to update.
        accum: The accumulation array.
        linear: The linear accumulation array.
        grad: The gradient array.
        **kwargs: Additional hyperparameters like lr.

    Returns:
        A tuple of (updated_param, updated_accum, updated_linear).
    """
    p, a, lin, g = np.asarray(param), np.asarray(accum), np.asarray(linear), np.asarray(grad)
    lr: object = kwargs.get("lr", 0.001)
    a_new: object = a + g**2
    sigma: object = (np.sqrt(a_new) - np.sqrt(a)) / lr
    l_new: object = lin - g + sigma * p
    p_new: object = p - lr * g
    return p_new, a_new, l_new


@numpy_eager_registry.register("ApplyRMSProp")
def _np_apply_rmsprop(backend_module: object, param: object, ms: object, mom: object, grad: object, **kwargs: object) -> tuple[object, ...]:
    """Numpy eager fallback for ApplyRMSProp.

    Args:
        backend_module: The active backend module.
        param: The parameter array to update.
        ms: The mean square accumulation array.
        mom: The momentum array.
        grad: The gradient array.
        **kwargs: Additional hyperparameters like lr.

    Returns:
        A tuple of (updated_param, updated_ms, updated_mom).
    """
    p, m, mo, g = np.asarray(param), np.asarray(ms), np.asarray(mom), np.asarray(grad)
    lr: object = kwargs.get("lr", 0.001)
    rho, momentum, eps = 0.9, 0.0, 1e-8
    m_new: object = m * rho + (g**2) * (1.0 - rho)
    mo_new: object = mo * momentum + g / (np.sqrt(m_new) + eps)
    p_new: object = p - lr * mo_new
    return p_new, m_new, mo_new
