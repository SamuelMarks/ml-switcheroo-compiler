"""Behavioral tests for NumPy eager optimizer steps (`ApplyAdam`, `ApplyAdagrad`, `ApplyFtrl`, `ApplyRMSProp`)."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.backends.eager.optimizers import apply_adagrad as _np_apply_adagrad
from ml_switcheroo_compiler.backends.eager.optimizers import apply_adam as _np_apply_adam
from ml_switcheroo_compiler.backends.eager.optimizers import apply_ftrl as _np_apply_ftrl
from ml_switcheroo_compiler.backends.eager.optimizers import apply_rmsprop as _np_apply_rmsprop
from ml_switcheroo_compiler.backends.eager_registry import coerce_operand_list, numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from tests.testing_utils import require_callable


def test_apply_adam_default_and_custom_lr() -> None:
    """Verify `ApplyAdam` computes exact first/second moment updates and parameter steps."""
    param = np.array([1.0, 2.0], dtype=np.float64)
    m = np.zeros_like(param)
    v = np.zeros_like(param)
    grad = np.array([0.1, 0.2], dtype=np.float64)

    default_out = coerce_operand_list(_np_apply_adam(np, param, m, v, grad))
    assert len(default_out) == 3
    expected_m = 0.1 * grad
    expected_v = 0.001 * (grad**2)
    expected_default_p = param - 0.001 * (expected_m / (np.sqrt(expected_v) + 1e-8))
    np.testing.assert_allclose(ensure_ndarray(default_out[0]), expected_default_p)
    np.testing.assert_allclose(ensure_ndarray(default_out[1]), expected_m)
    np.testing.assert_allclose(ensure_ndarray(default_out[2]), expected_v)

    custom_out = coerce_operand_list(_np_apply_adam(np, param, m, v, grad, lr=0.01))
    reg_adam = numpy_eager_registry.get("ApplyAdam")
    reg_out = require_callable(coerce_operand_list)(require_callable(reg_adam)(np, param, m, v, grad, lr=0.01))
    expected_custom_p = param - 0.01 * (expected_m / (np.sqrt(expected_v) + 1e-8))
    np.testing.assert_allclose(ensure_ndarray(custom_out[0]), expected_custom_p)
    np.testing.assert_allclose(ensure_ndarray(reg_out[0]), expected_custom_p)


def test_apply_adagrad_default_and_custom_lr() -> None:
    """Verify `ApplyAdagrad` accumulates squared gradients and scales parameter updates."""
    param = np.array([1.0, 2.0], dtype=np.float64)
    accum = np.zeros_like(param)
    grad = np.array([0.1, 0.2], dtype=np.float64)

    default_out = coerce_operand_list(_np_apply_adagrad(np, param, accum, grad))
    assert len(default_out) == 2
    expected_accum = grad**2
    expected_default_p = param - 0.01 * (grad / (np.sqrt(expected_accum) + 1e-10))
    np.testing.assert_allclose(ensure_ndarray(default_out[0]), expected_default_p)
    np.testing.assert_allclose(ensure_ndarray(default_out[1]), expected_accum)

    custom_out = coerce_operand_list(_np_apply_adagrad(np, param, accum, grad, lr=0.05))
    reg_adagrad = numpy_eager_registry.get("ApplyAdagrad")
    reg_out = require_callable(coerce_operand_list)(require_callable(reg_adagrad)(np, param, accum, grad, lr=0.05))
    expected_custom_p = param - 0.05 * (grad / (np.sqrt(expected_accum) + 1e-10))
    np.testing.assert_allclose(ensure_ndarray(custom_out[0]), expected_custom_p)
    np.testing.assert_allclose(ensure_ndarray(reg_out[0]), expected_custom_p)


def test_apply_ftrl_default_and_custom_lr() -> None:
    """Verify `ApplyFtrl` updates accumulator, linear state, and parameter tensor."""
    param = np.array([1.0, 2.0], dtype=np.float64)
    accum = np.zeros_like(param)
    linear = np.zeros_like(param)
    grad = np.array([0.1, 0.2], dtype=np.float64)

    default_out = coerce_operand_list(_np_apply_ftrl(np, param, accum, linear, grad))
    assert len(default_out) == 3
    expected_accum = grad**2
    expected_sigma_default = np.sqrt(expected_accum) / 0.001
    expected_linear_default = grad - expected_sigma_default * param
    np.testing.assert_allclose(ensure_ndarray(default_out[0]), param - 0.001, rtol=1e-5)
    np.testing.assert_allclose(ensure_ndarray(default_out[1]), expected_accum)
    np.testing.assert_allclose(ensure_ndarray(default_out[2]), expected_linear_default)

    custom_out = coerce_operand_list(_np_apply_ftrl(np, param, accum, linear, grad, lr=0.02))
    reg_ftrl = numpy_eager_registry.get("ApplyFtrl")
    reg_out = require_callable(coerce_operand_list)(require_callable(reg_ftrl)(np, param, accum, linear, grad, lr=0.02))
    np.testing.assert_allclose(ensure_ndarray(custom_out[0]), param - 0.02, rtol=1e-5)
    np.testing.assert_allclose(ensure_ndarray(reg_out[0]), param - 0.02, rtol=1e-5)


def test_apply_rmsprop_default_and_custom_lr() -> None:
    """Verify `ApplyRMSProp` updates mean-square state, momentum, and parameter tensor."""
    param = np.array([1.0, 2.0], dtype=np.float64)
    ms = np.ones_like(param)
    mom = np.zeros_like(param)
    grad = np.array([0.1, 0.2], dtype=np.float64)

    default_out = coerce_operand_list(_np_apply_rmsprop(np, param, ms, mom, grad))
    assert len(default_out) == 3
    expected_ms = 0.9 * ms + 0.1 * (grad**2)
    expected_mom = grad / (np.sqrt(expected_ms) + 1e-8)
    np.testing.assert_allclose(ensure_ndarray(default_out[0]), param - 0.001 * expected_mom)
    np.testing.assert_allclose(ensure_ndarray(default_out[1]), expected_ms)
    np.testing.assert_allclose(ensure_ndarray(default_out[2]), expected_mom)

    custom_out = coerce_operand_list(_np_apply_rmsprop(np, param, ms, mom, grad, lr=0.01))
    reg_rms = numpy_eager_registry.get("ApplyRMSProp")
    reg_out = require_callable(coerce_operand_list)(require_callable(reg_rms)(np, param, ms, mom, grad, lr=0.01))
    np.testing.assert_allclose(ensure_ndarray(custom_out[0]), param - 0.01 * expected_mom)
    np.testing.assert_allclose(ensure_ndarray(reg_out[0]), param - 0.01 * expected_mom)
