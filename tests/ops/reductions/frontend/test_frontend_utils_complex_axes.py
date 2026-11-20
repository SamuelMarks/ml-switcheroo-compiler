"""Tests for test_frontend_utils_complex_axes."""

from __future__ import annotations

import numpy as np
import pytest

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.configs import WindowConfig
from ml_switcheroo_compiler.ops.reductions.api_utils import ReduceWindow, reduce_window
from tests.testing_utils import MockTestStub


def test_frontend_utils_extras() -> None:
    """Test function test_frontend_utils_extras."""
    assert ReduceWindow().infer_shape() == ()

    dummy_cfg = WindowConfig(
        window_dimensions=(1, 1),
        window_strides=(1, 1),
        padding="VALID",
        base_dilation=(1, 1),
        window_dilation=(1, 1),
    )

    class MockData(MockTestStub):
        """Test class MockData."""

        id = "t1"

    class MockTensor(Tensor):
        """Test class MockTensor."""

        data = MockData()
        shape = (1, 2)
        dtype = DType("float32")
        device = Device("cpu")

    t = MockTensor(np.array([[1.0]]), TensorConfig(shape=(1, 2), dtype=DType("float32"), device=Device("cpu")))
    from ml_switcheroo_compiler.core.config import config

    orig = config.eager_mode
    config.eager_mode = False
    import ml_switcheroo_compiler.tracing.state as state

    orig_tracing = state.global_tracing_state.is_tracing
    state.global_tracing_state.is_tracing = True
    orig_add_node = state.global_tracing_state.add_node
    pytest.MonkeyPatch().setattr(state.global_tracing_state, "add_node", lambda node: None, raising=False)
    try:
        reduce_window(t, 0.0, "add", dummy_cfg)
    finally:
        config.eager_mode = orig
        state.global_tracing_state.is_tracing = orig_tracing
        pytest.MonkeyPatch().setattr(state.global_tracing_state, "add_node", orig_add_node, raising=False)
