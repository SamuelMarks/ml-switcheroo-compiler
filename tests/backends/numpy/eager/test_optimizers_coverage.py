"""Tests for test_optimizers_coverage."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import optimizers_ops as opt_module

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


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
    p_adam, m_adam, v_adam = cast("Callable[..., EagerTensorProtocol]", opt_module._np_apply_adam)(np, param, m, v, grad, lr=0.01)
    assert cast("EagerTensorProtocol", getattr(p_adam, "shape")) == cast("EagerTensorProtocol", getattr(param, "shape"))
    reg_adam = numpy_eager_registry.get("ApplyAdam")
    p_reg, _, _ = cast("Callable[..., EagerTensorProtocol]", reg_adam)(np, param, m, v, grad, lr=0.01)
    assert cast("Callable[..., EagerTensorProtocol]", np.allclose)(p_adam, p_reg)

    # Adagrad
    accum = np.zeros_like(param)
    p_adagrad, a_adagrad = cast("Callable[..., EagerTensorProtocol]", opt_module._np_apply_adagrad)(np, param, accum, grad, lr=0.05)
    assert cast("EagerTensorProtocol", getattr(p_adagrad, "shape")) == cast("EagerTensorProtocol", getattr(param, "shape"))
    reg_adagrad = numpy_eager_registry.get("ApplyAdagrad")
    p_ada_reg, _ = cast("Callable[..., EagerTensorProtocol]", reg_adagrad)(np, param, accum, grad, lr=0.05)
    assert cast("Callable[..., EagerTensorProtocol]", np.allclose)(p_adagrad, p_ada_reg)

    # Ftrl
    accum_ftrl = np.zeros_like(param)
    linear_ftrl = np.zeros_like(param)
    p_ftrl, a_ftrl, l_ftrl = cast("Callable[..., EagerTensorProtocol]", opt_module._np_apply_ftrl)(np, param, accum_ftrl, linear_ftrl, grad, lr=0.02)
    assert cast("EagerTensorProtocol", getattr(p_ftrl, "shape")) == cast("EagerTensorProtocol", getattr(param, "shape"))
    reg_ftrl = numpy_eager_registry.get("ApplyFtrl")
    p_ftrl_reg, _, _ = cast("Callable[..., EagerTensorProtocol]", reg_ftrl)(np, param, accum_ftrl, linear_ftrl, grad, lr=0.02)
    assert cast("Callable[..., EagerTensorProtocol]", np.allclose)(p_ftrl, p_ftrl_reg)

    # RMSProp
    ms = np.ones_like(param)
    mom = np.zeros_like(param)
    p_rms, m_rms, mo_rms = cast("Callable[..., EagerTensorProtocol]", opt_module._np_apply_rmsprop)(np, param, ms, mom, grad, lr=0.01)
    assert cast("EagerTensorProtocol", getattr(p_rms, "shape")) == cast("EagerTensorProtocol", getattr(param, "shape"))
    reg_rms = numpy_eager_registry.get("ApplyRMSProp")
    p_rms_reg, _, _ = cast("Callable[..., EagerTensorProtocol]", reg_rms)(np, param, ms, mom, grad, lr=0.01)
    assert cast("Callable[..., EagerTensorProtocol]", np.allclose)(p_rms, p_rms_reg)
