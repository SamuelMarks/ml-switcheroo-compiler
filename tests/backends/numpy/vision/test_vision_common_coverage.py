"""Tests for test_vision_common_coverage."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import vision_common as vision_common_module

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


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
    assert cast("EagerTensorProtocol", getattr(res_mel, "__getitem__")(0)) == 1.0
    reg_mel = numpy_eager_registry.get("MelFilterbank")
    assert cast("EagerTensorProtocol", getattr(cast("Callable[..., EagerTensorProtocol]", reg_mel)(np, None, config=None), "__getitem__")(0)) == 1.0

    res_mfcc = vision_common_module._np_mfcc(np, np.ones((4, 4)), config=None)
    assert cast("EagerTensorProtocol", getattr(res_mfcc, "__getitem__")(0)) == 2.0
    reg_mfcc = numpy_eager_registry.get("Mfcc")
    assert cast("EagerTensorProtocol", getattr(cast("Callable[..., EagerTensorProtocol]", reg_mfcc)(np, cast("Callable[..., EagerTensorProtocol]", np.ones)((4, 4)), config=None), "__getitem__")(0)) == 2.0

    # PowerIteration - branch 1: u is None
    w = np.eye(3, dtype=np.float32)
    v, u, sigma = cast("Callable[..., EagerTensorProtocol]", vision_common_module._np_power_iteration)(np, w, num_iters=2)
    assert cast("EagerTensorProtocol", getattr(v, "shape")) == (3,)
    assert cast("EagerTensorProtocol", getattr(u, "shape")) == (3,)

    # PowerIteration - branch 2: u is not None
    u_init = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2, u2, sigma2 = cast("Callable[..., EagerTensorProtocol]", vision_common_module._np_power_iteration)(np, w, u=u_init, num_iters=2)
    assert cast("EagerTensorProtocol", getattr(v2, "shape")) == (3,)
    assert cast("EagerTensorProtocol", getattr(u2, "shape")) == (3,)

    reg_pi = numpy_eager_registry.get("PowerIteration")
    v_reg, u_reg, sigma_reg = cast("Callable[..., EagerTensorProtocol]", reg_pi)(np, w, num_iters=2)
    assert cast("EagerTensorProtocol", getattr(v_reg, "shape")) == (3,)
