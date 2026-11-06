"""Tests for test_frontend_utils_complex_axes."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.reductions.frontend_utils import ReduceWindow, reduce_window
from tests.testing_utils import ensure_callable, ensure_eager_tensor, ensure_mutable

if typing.TYPE_CHECKING:
    pass


def test_frontend_utils_extras() -> None:
    """Test function test_frontend_utils_extras."""
    assert ReduceWindow().infer_shape() == ()

    class DummyConfig:
        """Test class DummyConfig."""

        window_dimensions = (1, 1)
        window_strides = (1, 1)
        padding = "VALID"
        base_dilation = (1, 1)
        window_dilation = (1, 1)

    class MockData:
        """Test class MockData."""

        id = "t1"

    class MockTensor(Tensor):
        """Test class MockTensor."""

        data = ensure_eager_tensor(MockData())
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
    ensure_mutable(state.global_tracing_state).add_node = ensure_eager_tensor(lambda node: None)
    try:
        ensure_callable(reduce_window)(t, 0.0, "add", ensure_callable(DummyConfig)())
    finally:
        config.eager_mode = orig
        state.global_tracing_state.is_tracing = orig_tracing
        ensure_mutable(state.global_tracing_state).add_node = ensure_eager_tensor(orig_add_node)
