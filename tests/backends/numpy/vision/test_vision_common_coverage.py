"""Tests for test_vision_common_coverage."""

from __future__ import annotations

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import vision_common as vision_common_module


def test_vision_common_ops(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test vision common operations directly and through eager registry.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for monkeypatching.

    Returns:
        None
    """
    for item in vision_common_module.__all__:
        assert hasattr(vision_common_module, item)

    monkeypatch.setattr(vision_common_module, "mel_filterbank_eager", lambda mod, spec, cfg: np.array([1.0]))
    monkeypatch.setattr(vision_common_module, "mfcc_eager", lambda mod, spec, cfg: np.array([2.0]))

    res_mel = vision_common_module._np_mel_filterbank(np, None, config=None)
    assert res_mel[0] == 1.0
    reg_mel = numpy_eager_registry.get("MelFilterbank")
    assert reg_mel(np, None, config=None)[0] == 1.0

    res_mfcc = vision_common_module._np_mfcc(np, np.ones((4, 4)), config=None)
    assert res_mfcc[0] == 2.0
    reg_mfcc = numpy_eager_registry.get("Mfcc")
    assert reg_mfcc(np, np.ones((4, 4)), config=None)[0] == 2.0

    # PowerIteration - branch 1: u is None
    w = np.eye(3, dtype=np.float32)
    v, u, sigma = vision_common_module._np_power_iteration(np, w, num_iters=2)
    assert v.shape == (3,)
    assert u.shape == (3,)

    # PowerIteration - branch 2: u is not None
    u_init = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2, u2, sigma2 = vision_common_module._np_power_iteration(np, w, u=u_init, num_iters=2)
    assert v2.shape == (3,)
    assert u2.shape == (3,)

    reg_pi = numpy_eager_registry.get("PowerIteration")
    v_reg, u_reg, sigma_reg = reg_pi(np, w, num_iters=2)
    assert v_reg.shape == (3,)
