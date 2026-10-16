"""Unit tests for Tensor and Variable methods and edge cases."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import numpy as np
import pytest

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig, Variable
from ml_switcheroo_compiler.tracing.state import global_tracing_state

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_tensor_coverage() -> None:
    """Test Tensor iteration, length, and indexing in eager mode."""
    config.eager_mode = True
    try:
        t = Tensor(np.array([1.0, 2.0]), TensorConfig((2,), DType.Float32, Device("cpu")))
        assert len(t) == 2
        elems = [x for x in t]
        assert len(elems) == 2
        assert t[0] is not None
    finally:
        config.eager_mode = False


def test_tensor_backward_and_view() -> None:
    """Test tensor view and backward methods."""
    config.eager_mode = True
    t = Tensor(np.array([2.0], dtype=np.float32), TensorConfig((1,), DType.Float32, "cpu"))
    t.backward()
    assert cast("EagerTensorProtocol", getattr(t.view(1), "shape")) == (1,)
    assert cast("EagerTensorProtocol", getattr(cast("Callable[..., EagerTensorProtocol]", t.view)([1]), "shape")) == (1,)


def test_tensor_more_methods() -> None:
    """Test contiguous, detach, int conversion, and Variable assign_add/assign_sub."""
    config.eager_mode = True
    t = Tensor(np.array(2, dtype=np.int32), TensorConfig((), DType.Int32, "cpu"))
    assert t.contiguous() is t
    assert t.detach() is not None
    assert int(t) == 2

    v = Variable(np.array(2.0, dtype=np.float32), TensorConfig((), DType.Float32, "cpu"))
    t_f = Tensor(np.array(1.0, dtype=np.float32), TensorConfig((), DType.Float32, "cpu"))
    assert v.assign_add(t_f) is not None
    assert v.assign_sub(t_f) is not None


def test_tensor_non_eager_and_index() -> None:
    """Test non-eager mode Variable and __index__."""
    v = Variable(np.array(2.0), TensorConfig((), DType.Float32, "cpu"))
    t = Tensor(np.array(1, dtype=np.int32), TensorConfig((), DType.Int32, "cpu"))
    old_eager = config.eager_mode
    config.eager_mode = False
    global_tracing_state.is_tracing = False
    try:
        with patch("ml_switcheroo_compiler.ops.registry.get_util", side_effect=RuntimeError("not tracing")):
            with pytest.raises(RuntimeError):
                v.assign_add(t)
            with pytest.raises(RuntimeError):
                v.assign_sub(t)
    finally:
        config.eager_mode = old_eager

    assert t.__index__() == 1


def test_tensor_config_list_shape() -> None:
    """Test initializing TensorConfig with list shape."""
    cfg = TensorConfig(shape=[2, 2], dtype=DType("float32"), device=Device("cpu"))
    assert cfg.shape == (2, 2)


def test_tensor_view() -> None:
    """Test tensor view with different shape args."""
    cfg = TensorConfig(shape=(2, 2), dtype=DType("float32"), device=Device("cpu"))
    t = Tensor(42, config=cfg)

    with patch("ml_switcheroo_compiler.ops.registry.get_frontend") as mock_frontend:
        mock_frontend.return_value = lambda x, s: x
        res = t.view(2, 2)
        assert res is t

        res2 = cast("Callable[..., EagerTensorProtocol]", t.view)([2, 2])
        assert id(res2) == id(t)


def test_valid_force_edge_cases() -> None:
    """Verify that edge inputs like zero-length arrays and empty shapes are handled correctly."""
    t1 = Tensor(np.array(0.0), TensorConfig((), DType.Float32, "cpu"))
    assert t1.shape == ()
    assert t1.data == 0.0

    t2 = Tensor(np.zeros((0, 5)), TensorConfig((0, 5), DType.Float32, "cpu"))
    assert t2.shape == (0, 5)
    assert cast("EagerTensorProtocol", getattr(t2.data, "size")) == 0
