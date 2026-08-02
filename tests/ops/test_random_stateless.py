"""Test random stateless."""

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.random_stateless import Generator, stateless_split
from ml_switcheroo_compiler.tracing import global_tracing_state


def test_random_stateless_tracing_coverage():
    """Test stateless random generator tracing modes."""
    sr = Generator(state=42)
    config.eager_mode = False
    global_tracing_state.start_tracing()
    try:
        t = Tensor(None, TensorConfig((2,), "int32", "cpu"))
        out_normal = sr.normal((3, 3))
        assert out_normal.shape == (3, 3)
        out_uniform = sr.uniform((4, 4))
        assert out_uniform.shape == (4, 4)
        out_split = stateless_split(t, 2)
        assert out_split.shape == (2, 2)
    finally:
        global_tracing_state.stop_tracing()
        config.eager_mode = True
