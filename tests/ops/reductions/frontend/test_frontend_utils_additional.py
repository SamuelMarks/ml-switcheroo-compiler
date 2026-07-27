import numpy as np

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.reductions.frontend_utils import ReduceWindow, reduce_window


def test_frontend_utils_extras():
    assert ReduceWindow().infer_shape() == ()

    class DummyConfig:
        window_dimensions = (1, 1)
        window_strides = (1, 1)
        padding = "VALID"
        base_dilation = (1, 1)
        window_dilation = (1, 1)

    class MockData:
        id = "t1"

    class MockTensor(Tensor):
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
    state.global_tracing_state.add_node = lambda node: None
    try:
        reduce_window(t, 0.0, "add", DummyConfig())
    finally:
        config.eager_mode = orig
        state.global_tracing_state.is_tracing = orig_tracing
        state.global_tracing_state.add_node = orig_add_node
