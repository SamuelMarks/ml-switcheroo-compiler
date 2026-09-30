"""Tests for test_optimizers_coverage."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import optimizers_ops as opt_module


def test_optimizers_ops() -> None:
    """Test optimizer step operations directly and through eager registry.

    Returns:
        None
    """
    param = np.array([1.0, 2.0], dtype=np.float32)
    grad = np.array([0.1, 0.2], dtype=np.float32)

    # Adam
    m = np.zeros_like(param)
    v = np.zeros_like(param)
    p_adam, m_adam, v_adam = opt_module._np_apply_adam(np, param, m, v, grad, lr=0.01)
    assert p_adam.shape == param.shape
    reg_adam = numpy_eager_registry.get("ApplyAdam")
    p_reg, _, _ = reg_adam(np, param, m, v, grad, lr=0.01)
    assert np.allclose(p_adam, p_reg)

    # Adagrad
    accum = np.zeros_like(param)
    p_adagrad, a_adagrad = opt_module._np_apply_adagrad(np, param, accum, grad, lr=0.05)
    assert p_adagrad.shape == param.shape
    reg_adagrad = numpy_eager_registry.get("ApplyAdagrad")
    p_ada_reg, _ = reg_adagrad(np, param, accum, grad, lr=0.05)
    assert np.allclose(p_adagrad, p_ada_reg)

    # Ftrl
    accum_ftrl = np.zeros_like(param)
    linear_ftrl = np.zeros_like(param)
    p_ftrl, a_ftrl, l_ftrl = opt_module._np_apply_ftrl(np, param, accum_ftrl, linear_ftrl, grad, lr=0.02)
    assert p_ftrl.shape == param.shape
    reg_ftrl = numpy_eager_registry.get("ApplyFtrl")
    p_ftrl_reg, _, _ = reg_ftrl(np, param, accum_ftrl, linear_ftrl, grad, lr=0.02)
    assert np.allclose(p_ftrl, p_ftrl_reg)

    # RMSProp
    ms = np.ones_like(param)
    mom = np.zeros_like(param)
    p_rms, m_rms, mo_rms = opt_module._np_apply_rmsprop(np, param, ms, mom, grad, lr=0.01)
    assert p_rms.shape == param.shape
    reg_rms = numpy_eager_registry.get("ApplyRMSProp")
    p_rms_reg, _, _ = reg_rms(np, param, ms, mom, grad, lr=0.01)
    assert np.allclose(p_rms, p_rms_reg)
