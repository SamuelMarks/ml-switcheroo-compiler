"""Tests for test_adamw."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

import ml_switcheroo_compiler.ops.optimizers.updates as upd
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol


def test_adamw_no_weight_decay() -> None:
    """Test function test_adamw_no_weight_decay."""
    orig = config.eager_mode
    config.eager_mode = True
    try:
        t = Tensor(np.array([1.0]), TensorConfig(shape=(1,), dtype=DType("float32"), device=Device("cpu")))
        hp = upd.AdamWHyperparams(lr=0.1, weight_decay=0.0)
        upd.adamw_update(t, t, hp)
    finally:
        config.eager_mode = orig
    assert config.eager_mode == orig


def test_missing_optimizers_coverage() -> None:
    """Test function test_missing_optimizers_coverage."""
    from ml_switcheroo_compiler.ops.optimizers.updates import AdamaxHyperparamsOp, ApplyRMSProp

    class DummyShape:
        """Test class DummyShape."""

        shape = (1,)

    with pytest.raises(TypeError):
        cast("EagerCallableProtocol", ApplyRMSProp)().infer_shape(cast("EagerCallableProtocol", DummyShape)())
    cast("EagerCallableProtocol", AdamaxHyperparamsOp)().infer_shape(cast("EagerCallableProtocol", DummyShape)())
