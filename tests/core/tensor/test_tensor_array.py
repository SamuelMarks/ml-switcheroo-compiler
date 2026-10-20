"""Test module."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.core.tensor_array import TensorArray

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def test_tensor_array() -> None:
    """Test function test_tensor_array."""
    ta = TensorArray(10, (2, 2), "float32")
    assert ta.size == 10
    assert ta.element_shape == (2, 2)
    assert ta.dtype == "float32"

    import numpy as np

    class DummyData:
        """Test class DummyData."""

        def __init__(self, val: EagerValue = 0) -> None:
            """Test function __init__.

            Args:
                val: Test parameter `val`.
            """
            self.id = "id"
            self.val = val

        def __int__(self) -> EagerValue:
            """Test function __int__."""
            return self.val

        def __array__(self, dtype: EagerValue = None) -> EagerValue:
            """Test function __array__.

            Args:
                dtype: Test parameter `dtype`.
            """
            return np.array(self.val)

    t_idx = Tensor(data="test", config=TensorConfig((1,), "int32", "cpu"))
    setattr(t_idx, "_data", DummyData(0))

    t_val = Tensor(data="val", config=TensorConfig((2, 2), "float32", "cpu"))
    t_val._data = np.ones((2, 2), dtype="float32")

    import ml_switcheroo_compiler.tracing.state as state

    state.global_tracing_state.is_tracing = False

    r = ta.read(t_idx)
    assert r.shape == (2, 2)

    ta2 = ta.write(t_idx, t_val)
    assert ta2 is ta

    s = ta.stack()
    assert s.shape == (10, 2, 2)

    # Check tracing
    state.global_tracing_state.start_tracing("test_tensor_array")
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

    config.eager_mode = False

    t_idx_proxy = Tensor(ProxyTensor(id="idx", shape=(), dtype="int32"), TensorConfig((), "int32", "cpu"))
    t_val_proxy = Tensor(ProxyTensor(id="val", shape=(2, 2), dtype="float32"), TensorConfig((2, 2), "float32", "cpu"))

    r = ta.read(t_idx_proxy)
    ta.write(t_idx_proxy, t_val_proxy)
    s = ta.stack()

    assert len(typing.cast("EagerTensorProtocol", getattr(state.global_tracing_state.active_graph, "nodes"))) >= 3
    state.global_tracing_state.is_tracing = False
